"""Frozen-design and artifact utilities, safe on the Mac (no model imports)."""
import hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SEED=20261025

def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')

def readrows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines()] if Path(path).exists() else []

def frozen():
    for manifest in ['frozen-sha256.json','amendment-sha256.json','source-amendment-sha256.json']:
        for p,sha in json.loads((ROOT/'configs'/manifest).read_text()).items():
            assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha,p

def conditions(shifted=False):
    cs=json.loads((ROOT/'configs/conditions.json').read_text())+json.loads((ROOT/'configs/rotated-conditions.json').read_text())
    if shifted:cs+=json.loads((ROOT/'configs/indexing-conditions.json').read_text())
    return cs

def assert_main_ready(validation=True):
    """A pilot scalar or stale validation must never silently become a main run."""
    provenance=json.loads((ROOT/'runs/calibration-provenance.json').read_text())
    assert provenance['tokens_processed']>=2_000_000,'source calibration incomplete'
    actual=hashlib.sha256((ROOT/'runs/calibration.json').read_bytes()).hexdigest()
    assert provenance['calibration_sha256']==actual,'calibration provenance mismatch'
    if validation:
        verified=json.loads((ROOT/'runs/validation.json').read_text())
        assert verified['calibration_sha256']==actual,'stale or pilot-only validation'
