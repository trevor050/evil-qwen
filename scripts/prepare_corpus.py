"""Download two fixed shards, then tokenize locally in a separate bounded process."""
import argparse,hashlib,json,os,random,subprocess,time
from pathlib import Path
import numpy as np
from experiment import ROOT,write

SHARDS=[
    ('web','HuggingFaceFW/fineweb-edu','87f09149ef4734204d70ed1d046ddc9ca3f2b8f9','sample/10BT/013_00000.parquet'),
    ('chat','HuggingFaceH4/ultrachat_200k','8049631c405ae6576f93f445c6b8166f76f5505a','data/train_sft-00000-of-00003-a3ecf92756993583.parquet')]

def evict(path):
    if hasattr(os,'posix_fadvise'):
        with open(path,'rb') as f:os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    evict(path);return h.hexdigest()

def download():
    folder=ROOT/'runs/corpus-shards';folder.mkdir(parents=True,exist_ok=True);rows=[]
    for name,repo,rev,path in SHARDS:
        dest=folder/(name+'.parquet');url=f'https://huggingface.co/datasets/{repo}/resolve/{rev}/{path}?download=true'
        if not dest.exists():
            part=dest.with_suffix('.part')
            subprocess.run(['curl','--fail','--location','--retry','3','--continue-at','-','--limit-rate','20M','--output',str(part),url],check=True)
            part.rename(dest)
        evict(dest)
        rows.append(dict(name=name,repository=repo,revision=rev,path=path,bytes=dest.stat().st_size,sha256=sha(dest)))
        print(json.dumps(dict(stage='download-complete',**rows[-1])),flush=True)
    write(ROOT/'runs/corpus-shards.json',rows)

def rows(path,column,rng):
    import pyarrow.parquet as pq
    reader=pq.ParquetFile(path);buffer=[]
    for batch in reader.iter_batches(batch_size=16,columns=[column],use_threads=False):
        for row in batch.to_pylist():
            buffer.append(row[column])
            if len(buffer)>=128:yield buffer.pop(rng.randrange(len(buffer)))
    while buffer:yield buffer.pop(rng.randrange(len(buffer)))
    reader.close();evict(path)

def tokenize():
    from transformers import AutoTokenizer
    from run import MODEL
    tok=AutoTokenizer.from_pretrained(MODEL);folder=ROOT/'runs/corpus-shards'
    web=rows(folder/'web.parquet','text',random.Random(0));chat=rows(folder/'chat.parquet','messages',random.Random(0))
    target=2_002_944;done=0;t=time.time();counts=dict(web=0,chat=0);last=0
    path=ROOT/'runs/calibration-inputs.i32';part=path.with_suffix('.part')
    with part.open('wb') as f:
        for w,c in zip(web,chat):
            for name,text in [('web',w),('chat',tok.apply_chat_template(c,tokenize=False,enable_thinking=False))]:
                ids=tok(text,add_special_tokens=False)['input_ids']+[tok.eos_token_id]
                ids=ids[:target-done];f.write(np.asarray(ids,dtype='<i4').tobytes());done+=len(ids);counts[name]+=len(ids)
                if time.time()-last>=15:
                    print(json.dumps(dict(stage='tokenize',tokens=done,elapsed=time.time()-t)),flush=True);last=time.time()
                if done>=target:break
            if done>=target:break
    assert done>=target;part.rename(path);evict(path)
    write(ROOT/'runs/calibration-inputs.json',dict(tokens=done,sha256=sha(path),source_token_counts=counts,seed=0,
        source='local fixed shards, alternating web/chat documents, EOS separated, shuffle buffer128, 1024-token windows',
        shards=json.loads((ROOT/'runs/corpus-shards.json').read_text())))
    for path in folder.iterdir():evict(path)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['download','tokenize']);a=p.parse_args();os.chdir(ROOT)
    {'download':download,'tokenize':tokenize}[a.stage]()
