"""Bound each job's aggregate RSS; emit durable 30s liveness outside WSL."""
import argparse,json,os,signal,subprocess,time
from pathlib import Path
import psutil

def main():
    p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    external=Path('/mnt/c/Users/Trevor/AppData/Local/Temp/evilqwen-progress');external.mkdir(parents=True,exist_ok=True)
    child=subprocess.Popen(a.command,start_new_session=True);start=time.time();last=0
    while child.poll() is None:
        try:
            proc=psutil.Process(child.pid);rss=sum(x.memory_info().rss for x in [proc]+proc.children(recursive=True) if x.is_running())
        except psutil.Error:rss=0
        mem=psutil.virtual_memory()
        if rss>9*1024**3 or mem.available<2200*1024**2:
            event=dict(stage='memory_guard',job=a.name,rss=rss,available=mem.available,elapsed=time.time()-start)
            print(json.dumps(event),flush=True);(external/(a.name+'.json')).write_text(json.dumps(event))
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
            raise SystemExit(90)
        if time.time()-last>=30:
            event=dict(stage='heartbeat',job=a.name,pid=child.pid,elapsed_seconds=round(time.time()-start,1),rss_mb=round(rss/1024**2,1),
                free_m=subprocess.check_output(['free','-m'],text=True))
            progress=Path('runs/progress.json')
            if progress.exists():
                try:event['progress']=json.loads(progress.read_text())
                except ValueError:pass
            print(json.dumps(event),flush=True);(external/(a.name+'.json')).write_text(json.dumps(event));last=time.time()
        time.sleep(2)
    event=dict(stage='job_exit',job=a.name,returncode=child.returncode,elapsed_seconds=round(time.time()-start,1))
    print(json.dumps(event),flush=True);(external/(a.name+'.json')).write_text(json.dumps(event))
    raise SystemExit(child.returncode)

if __name__=='__main__':main()
