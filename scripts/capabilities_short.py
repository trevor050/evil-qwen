"""Amended last GPU stage: first50 questions, 13 conditions, <=25min."""
import argparse,json,os,subprocess,sys,time
import torch
from experiment import ROOT,frozen,assert_main_ready,readrows,write
from run import Harness,append,number
from judge import opaque

def wait_and_publish():
    expected={(x['condition'],x['id']) for x in readrows(ROOT/'runs/behavior.jsonl')}
    assert len(expected)==3060
    expected_judge={opaque(c,p) for c,p in expected}
    while not expected_judge<={x['id'] for x in readrows(ROOT/'runs/judgments.jsonl')}:
        write(ROOT/'runs/progress.json',dict(stage='waiting-for-final-blind-judging',gpu_loaded=False))
        time.sleep(2)
    subprocess.run([sys.executable,'scripts/analyze.py','--partial'],check=True)
    subprocess.run([sys.executable,'scripts/write_results.py','--pending-capabilities'],check=True)
    write(ROOT/'runs/behavior-report-ready.json',dict(unix_time=time.time(),capability_status='pending',judgments=3060))

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',type=int,default=8);a=p.parse_args()
    os.chdir(ROOT);frozen();assert_main_ready();wait_and_publish()
    design=json.loads((ROOT/'configs/capability-short.json').read_text())
    h=Harness();h.cal=json.loads((ROOT/'runs/calibration.json').read_text())
    samples={bench:json.loads((ROOT/'runs'/file).read_text())[:50] for bench,file in [('gsm','gsm100.json'),('mmlu','mmlu100.json')]}
    for bench,rs in samples.items():assert [x['id'] for x in rs]==design['questions'][bench]
    path=ROOT/'runs/capabilities.jsonl';done={(x['condition'],x['id']) for x in readrows(path)};completed=len(done)
    choices=[h.tok.encode(x,add_special_tokens=False) for x in 'ABCD'];assert all(len(x)==1 for x in choices)
    for c in design['conditions']:
        for bench,rows in samples.items():
            for start in range(0,len(rows),a.batch):
                batch=[x for x in rows[start:start+a.batch] if (c['id'],x['id']) not in done]
                if not batch:continue
                t=time.time()
                if bench=='gsm':
                    rs=h.generate([x['question']+'\nGive your solution and end with #### followed by the numeric answer.' for x in batch],c['neurons'],rotation_seed=c.get('rotation_seed'))
                    for x,r in zip(batch,rs):append(path,dict(id=x['id'],benchmark=bench,condition=c['id'],**r,prediction=number(r['response']),answer=number(x['answer']),correct=number(r['response'])==number(x['answer'])))
                else:
                    prompts=[x['question']+'\n'+'\n'.join(f'{k}. {v}' for k,v in zip('ABCD',x['choices']))+'\nAnswer with the letter of the correct option only.' for x in batch]
                    with h.steer(c['neurons'],c.get('rotation_seed')):logits=h.model(**h.encode(prompts),use_cache=False,logits_to_keep=1).logits[:,-1].float()
                    scores=logits[:,[x[0] for x in choices]].cpu().numpy()
                    for x,s in zip(batch,scores):append(path,dict(id=x['id'],benchmark=bench,condition=c['id'],prediction=int(s.argmax()),correct=bool(s.argmax()==x['answer']),choice_logits=s.tolist()))
                completed+=len(batch);event=dict(stage='capabilities-short',benchmark=bench,condition=c['id'],completed_rows=completed,total=1300,seconds=time.time()-t)
                write(ROOT/'runs/progress.json',event);print(json.dumps(event),flush=True)

if __name__=='__main__':main()
