"""Qwen3.5 text-only, static calibrated additive MLP-neuron steering."""
import argparse, contextlib, hashlib, json, os, platform, random, re, time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, Qwen3_5ForCausalLM

ROOT = Path(__file__).resolve().parents[1]
MODEL = '/home/user/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
SEED = 20261025

def write(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False)+'\n')

def frozen():
    for manifest in ['frozen-sha256.json','amendment-sha256.json']:
        for p, sha in json.loads((ROOT/'configs'/manifest).read_text()).items():
            assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest() == sha, p

def conditions():
    return json.loads((ROOT/'configs/conditions.json').read_text()) + json.loads((ROOT/'configs/indexing-conditions.json').read_text())

def kernel_setup():
    # Explicit replacements preserve the HF function signatures and mathematical operation.
    import transformers.models.qwen3_5.modeling_qwen3_5 as q
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
    q.torch_chunk_gated_delta_rule = chunk_gated_delta_rule
    q.torch_recurrent_gated_delta_rule = fused_recurrent_gated_delta_rule
    return 'fla chunk_gated_delta_rule + fused_recurrent_gated_delta_rule; HF torch conv1d'

class Harness:
    def __init__(self, use_fla=True):
        torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
        torch.set_num_threads(6)
        self.kernel = kernel_setup() if use_fla else 'HF torch fallback'
        t = time.time()
        self.tok = AutoTokenizer.from_pretrained(MODEL, padding_side='left')
        self.model = Qwen3_5ForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map='cuda', attn_implementation='sdpa').eval()
        self.layers = self.model.model.layers
        assert len(self.layers)==32 and self.layers[14].mlp.down_proj.weight.shape==(2560,9216)
        self.cal = None
        write(ROOT/'runs/environment.json', dict(torch=torch.__version__, transformers=__import__('transformers').__version__,
            python=platform.python_version(), model=MODEL, kernel=self.kernel, seed=SEED, loaded_seconds=time.time()-t,
            gpu=torch.cuda.get_device_name(), allocated_gb=torch.cuda.memory_allocated()/1e9,
            chat_template_sha256=hashlib.sha256(self.tok.chat_template.encode()).hexdigest()))

    def encode(self, prompts, thinking=False):
        texts=[self.tok.apply_chat_template([{'role':'user','content':p}], tokenize=False,
            add_generation_prompt=True, enable_thinking=thinking) for p in prompts]
        return self.tok(texts, padding=True, return_tensors='pt').to('cuda')

    @torch.inference_mode()
    def calibrate(self):
        norms={13:[],14:[],15:[]}
        for prompt in json.loads((ROOT/'configs/calibration.json').read_text()):
            hs=[]
            def hook(layer):
                def f(m, inp, out):
                    x=out[0] if isinstance(out,tuple) else out
                    norms[layer].extend(x[0,1:].float().norm(dim=-1).cpu().tolist())
                return f
            for layer in norms: hs.append(self.layers[layer].register_forward_hook(hook(layer)))
            try: self.model(**self.encode([prompt]), use_cache=False, logits_to_keep=1)
            finally:
                for h in hs: h.remove()
        self.cal={str(k):dict(median_norm=float(np.median(v)), token_count=len(v), min=min(v), max=max(v)) for k,v in norms.items()}
        write(ROOT/'runs/calibration.json',self.cal)
        return self.cal

    @contextlib.contextmanager
    def steer(self, neurons):
        handles=[]
        grouped={}
        for layer,n,dose in neurons: grouped.setdefault(layer,[]).append((n,dose))
        for layer, edits in grouped.items():
            proj=self.layers[layer].mlp.down_proj
            offsets=[(n, float(d)*self.cal[str(layer)]['median_norm']/float(proj.weight[:,n].float().norm())) for n,d in edits]
            def hook(m, inputs, offsets=offsets):
                # Return the original tensor for an exact no-op at dose zero.
                if not any(a for _,a in offsets): return None
                x=inputs[0].clone()
                for n,a in offsets: x[...,n] += a
                return (x,)+inputs[1:]
            handles.append(proj.register_forward_pre_hook(hook))
        try: yield
        finally:
            for h in handles: h.remove()

    @torch.inference_mode()
    def generate(self, prompts, neurons, max_tokens=192):
        inputs=self.encode(prompts)
        with self.steer(neurons):
            out=self.model.generate(**inputs, do_sample=False, max_new_tokens=max_tokens, use_cache=True,
                pad_token_id=self.tok.pad_token_id, return_dict_in_generate=True)
        gen=out.sequences[:,inputs.input_ids.shape[1]:].cpu().tolist()
        eos=self.model.generation_config.eos_token_id
        eos=set(eos if isinstance(eos,list) else [eos])
        results=[]
        for ids in gen:
            end=next((i+1 for i,x in enumerate(ids) if x in eos),len(ids));ids=ids[:end]
            results.append(dict(response=self.tok.decode(ids,skip_special_tokens=True),generated_tokens=len(ids),
                stopped_eos=bool(ids and ids[-1] in eos), token_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest()))
        return results

    @torch.inference_mode()
    def validate(self):
        prompt='What is 2 plus 2?'
        inputs=self.encode([prompt])
        a=self.model(**inputs,use_cache=False,logits_to_keep=1).logits
        with self.steer([[14,8949,0],[13,7956,0]]):
            b=self.model(**inputs,use_cache=False,logits_to_keep=1).logits
        assert torch.equal(a,b), 'dose zero changed logits'
        ga=self.generate([prompt],[],24)
        gb=self.generate([prompt],[[14,8949,0],[13,7956,0]],24)
        assert ga==gb, 'dose zero changed tokens'
        tests=[]
        for layer,n,d in [[14,8949,1],[13,7956,-1],[14,8949,.25]]:
            w=self.layers[layer].mlp.down_proj.weight[:,n].float()
            target=abs(d)*self.cal[str(layer)]['median_norm'];a=d*self.cal[str(layer)]['median_norm']/float(w.norm())
            norm=float((w*a).norm()); assert abs(norm-target)/target<1e-5
            # Actual bf16 projection difference includes input and GEMM rounding.
            proj=self.layers[layer].mlp.down_proj
            x=torch.zeros(1,2,9216,dtype=torch.bfloat16,device='cuda')
            y0=proj(x)
            with self.steer([[layer,n,d]]): y1=proj(x)
            observed=float((y1-y0).float().norm(dim=-1).mean())
            assert abs(observed-target)/target<.015
            tests.append(dict(layer=layer,neuron=n,dose=d,activation_offset=a,target_norm=target,fp32_norm=norm,bf16_observed_norm=observed))
        # Confirm prompt and decode token positions all pass through the pre-hook.
        calls=[]
        h=self.layers[14].mlp.down_proj.register_forward_pre_hook(lambda m,inp:calls.append(list(inp[0].shape)))
        try: self.generate([prompt],[[14,8949,1]],8)
        finally: h.remove()
        assert calls[0][1]>1 and any(s[1]==1 for s in calls[1:])
        write(ROOT/'runs/validation.json',dict(dose_zero_logits_exact=True,dose_zero_generations_exact=True,
            injection_norms=tests,prompt_and_decode_hook_shapes=calls))

def append(path,row):
    with open(path,'a') as f: f.write(json.dumps(row,ensure_ascii=False)+'\n'); f.flush(); os.fsync(f.fileno())

def readrows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines()] if Path(path).exists() else []

def behavior(h,args):
    prompts=json.loads((ROOT/'configs/prompts.json').read_text())
    conds=conditions()
    if args.conditions: conds=[x for x in conds if x['id'] in args.conditions.split(',')]
    path=ROOT/'runs/behavior.jsonl'; done={(x['condition'],x['id']) for x in readrows(path)}
    for c in conds:
        remaining=[p for p in prompts if (c['id'],p['id']) not in done]
        for start in range(0,len(remaining),args.batch):
            ps=remaining[start:start+args.batch]; t=time.time()
            rs=h.generate([p['prompt'] for p in ps],c['neurons'])
            for p,r in zip(ps,rs): append(path,dict(**p,**r,condition=c['id'],seconds_batch=time.time()-t,batch_size=len(ps)))
            print(json.dumps(dict(stage='behavior',condition=c['id'],finished=start+len(ps),total=len(remaining),seconds=time.time()-t)),flush=True)

def prepare_capabilities():
    from datasets import load_dataset
    gsm=load_dataset('openai/gsm8k','main',split='test')
    mmlu=load_dataset('cais/mmlu','all',split='test')
    rng=random.Random(SEED)
    gs=[dict(id=f'gsm_{i}',source_index=i,question=gsm[i]['question'],answer=gsm[i]['answer']) for i in rng.sample(range(len(gsm)),100)]
    ms=[dict(id=f'mmlu_{i}',source_index=i,**{k:mmlu[i][k] for k in ['question','choices','answer','subject']}) for i in rng.sample(range(len(mmlu)),100)]
    write(ROOT/'runs/gsm100.json',gs);write(ROOT/'runs/mmlu100.json',ms)
    write(ROOT/'runs/dataset-provenance.json',dict(gsm_fingerprint=gsm._fingerprint,mmlu_fingerprint=mmlu._fingerprint,
        gsm_info=str(gsm.info),mmlu_info=str(mmlu.info),seed=SEED))

def number(text):
    # Prefer the requested final-answer marker; fallback last numeric span.
    part=text.rsplit('####',1)[-1]
    matches=re.findall(r'-?\d[\d,]*(?:\.\d+)?',part)
    if not matches:return None
    try:return float(matches[-1].replace(',',''))
    except ValueError:return None

@torch.inference_mode()
def capabilities(h,args):
    conds=[x for x in conditions() if x['id'] in ['baseline','pair_1','shift_pair_1'] or 'random_' in x['id']]
    if args.conditions:conds=[x for x in conds if x['id'] in args.conditions.split(',')]
    path=ROOT/'runs/capabilities.jsonl';done={(x['condition'],x['id']) for x in readrows(path)}
    gs=json.loads((ROOT/'runs/gsm100.json').read_text());ms=json.loads((ROOT/'runs/mmlu100.json').read_text())
    choices=[h.tok.encode(x,add_special_tokens=False) for x in 'ABCD']
    assert all(len(x)==1 for x in choices),choices
    for c in conds:
        if args.benchmark in ('both','mmlu'):
            for start in range(0,len(ms),args.batch):
                batch=[x for x in ms[start:start+args.batch] if (c['id'],x['id']) not in done]
                if not batch:continue
                prompts=[x['question']+'\n'+'\n'.join(f'{k}. {v}' for k,v in zip('ABCD',x['choices']))+'\nAnswer with the letter of the correct option only.' for x in batch]
                with h.steer(c['neurons']): logits=h.model(**h.encode(prompts),use_cache=False,logits_to_keep=1).logits[:,-1].float()
                scores=logits[:,[x[0] for x in choices]].cpu().numpy()
                for x,s in zip(batch,scores):append(path,dict(id=x['id'],benchmark='mmlu',condition=c['id'],prediction=int(s.argmax()),correct=bool(s.argmax()==x['answer']),choice_logits=s.tolist()))
            print(json.dumps(dict(stage='mmlu',condition=c['id'])),flush=True)
        if args.benchmark in ('both','gsm'):
            for start in range(0,len(gs),args.batch):
                batch=[x for x in gs[start:start+args.batch] if (c['id'],x['id']) not in done]
                if not batch:continue
                t=time.time();rs=h.generate([x['question']+'\nGive your solution and end with #### followed by the numeric answer.' for x in batch],c['neurons'])
                for x,r in zip(batch,rs):append(path,dict(id=x['id'],benchmark='gsm',condition=c['id'],**r,prediction=number(r['response']),answer=number(x['answer']),correct=number(r['response'])==number(x['answer'])))
                print(json.dumps(dict(stage='gsm',condition=c['id'],finished=start+len(batch),seconds=time.time()-t)),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['calibrate','behavior','capabilities','prepare-capabilities']);p.add_argument('--batch',type=int,default=8);p.add_argument('--conditions');p.add_argument('--benchmark',choices=['both','mmlu','gsm'],default='both');p.add_argument('--torch-fallback',action='store_true');args=p.parse_args()
    os.chdir(ROOT);frozen();(ROOT/'runs').mkdir(exist_ok=True)
    if args.stage=='prepare-capabilities':prepare_capabilities();return
    h=Harness(not args.torch_fallback)
    if args.stage=='calibrate':h.calibrate();h.validate()
    else:
        h.cal=json.loads((ROOT/'runs/calibration.json').read_text())
        {'behavior':behavior,'capabilities':capabilities}[args.stage](h,args)

if __name__=='__main__':main()
