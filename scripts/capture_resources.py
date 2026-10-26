"""Copy liveness logs and summarize recorded supervisor exit statistics."""
import json,shutil
from pathlib import Path
from experiment import ROOT,write

def capture():
    dest=ROOT/'runs/job-logs';dest.mkdir(parents=True,exist_ok=True);summary={}
    for source in sorted((Path.home()/'evilqwen/logs').glob('*.log')):
        shutil.copy2(source,dest/source.name)
        for line in source.read_text(errors='replace').splitlines():
            try:x=json.loads(line)
            except ValueError:continue
            if x.get('stage')=='job_exit':summary[x['job']]=x
    write(ROOT/'results/resource-summary.json',summary)
    return summary

if __name__=='__main__':capture()
