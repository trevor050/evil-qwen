import json,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'scripts'))
from judge import validate_ratings
r=root/'runs';path=r/'judgments.jsonl';done={json.loads(x)['id'] for x in path.read_text().splitlines()};recovered=[]
for msg in json.loads((r/'judge-failures.json').read_text()):
 folder=Path(msg.split('Judge failed: ')[-1]);inputs=json.loads((folder/'input.json').read_text())
 for output in sorted(folder.glob('output-*.json')):
  try:ratings=validate_ratings(inputs,json.loads(output.read_text())['ratings'])
  except (AssertionError,ValueError,KeyError):continue
  added=[x for x in ratings if x['id'] not in done]
  with path.open('a') as f:
   for x in added:f.write(json.dumps(x,ensure_ascii=False)+'\n');done.add(x['id'])
  recovered.append(dict(source=str(output.relative_to(root)),added_ids=[x['id'] for x in added]));break
(r/'judge-scope-recovery.json').write_text(json.dumps(recovered,indent=2)+'\n')
print(json.dumps(dict(total=len(done),recovered=recovered)))
