"""Stream-hash model inputs and record source code, without retaining file cache."""
import hashlib,os,subprocess
from pathlib import Path
from experiment import ROOT,write

model=Path('/home/user/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a')
hashes={}
for p in sorted(model.iterdir()):
    if not p.is_file():continue
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
        if hasattr(os,'posix_fadvise'):os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
    hashes[p.name]=dict(bytes=p.stat().st_size,sha256=h.hexdigest())
    print(p.name,p.stat().st_size,flush=True)
write(ROOT/'runs/model-file-sha256.json',hashes)
write(ROOT/'runs/script-sha256.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'scripts').iterdir()) if p.is_file()})
