"""Mac transport only; no model or experiment code executes on the Mac."""
import argparse, shlex, subprocess, tarfile, tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
WRAPPER=Path.home()/'.codex/skills/remote-windows-pc/scripts/remote_pc.py'
DEST='C:/Users/Trevor/AppData/Local/Temp/evilqwen-code.tar.gz'

def call(*args):subprocess.run(['python3',str(WRAPPER),*args],check=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['sync','exec','fetch']);p.add_argument('args',nargs='*');a=p.parse_args()
    if a.action=='sync':
        with tempfile.TemporaryDirectory(prefix='evilqwen-') as d:
            archive=Path(d)/'code.tar.gz'
            with tarfile.open(archive,'w:gz') as t:
                for name in ['scripts','configs','AGENTS.md','README.md','requirements.in','requirements-lock.txt']:
                    if (ROOT/name).exists():t.add(ROOT/name,arcname=name,filter=lambda i:None if '__pycache__' in i.name else i)
            call('upload',str(archive),DEST)
        call('wsl','--command','mkdir -p ~/codeprojects/evil-qwen; cd ~/codeprojects/evil-qwen && tar -xzf /mnt/c/Users/Trevor/AppData/Local/Temp/evilqwen-code.tar.gz')
    elif a.action=='exec':
        command='cd /home/user/codeprojects/evil-qwen && '+shlex.join(a.args)
        call('wsl','--command',command)
    else:
        call('wsl','--command','cd ~/codeprojects/evil-qwen && tar -czf /mnt/c/Users/Trevor/AppData/Local/Temp/evilqwen-results.tar.gz runs results')
        archive=ROOT/'runs/remote-results.tar.gz';archive.parent.mkdir(exist_ok=True)
        call('download','C:/Users/Trevor/AppData/Local/Temp/evilqwen-results.tar.gz',str(archive))
        # Only locally generated experiment artifacts from the fixed remote checkout.
        with tarfile.open(archive) as t:
            t.extractall(ROOT,filter='data')

if __name__=='__main__':main()
