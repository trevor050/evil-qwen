"""Synthetic evidence tests only, never a model run or empirical result."""
import contextlib,hashlib,io,json,shutil,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import analyze,experiment
from judge import opaque,validate_ratings

class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='evilqwen-analysis-test-');self.root=Path(self.tmp.name)
        shutil.copytree(experiment.ROOT/'configs',self.root/'configs');(self.root/'runs').mkdir()
        self.patches=[patch.object(experiment,'ROOT',self.root),patch.object(analyze,'ROOT',self.root)]
        for p in self.patches:p.start()

    def tearDown(self):
        for p in self.patches:p.stop()
        self.tmp.cleanup()

    def test_pilot_calibration_cannot_authorize_main(self):
        experiment.write(self.root/'runs/calibration.json',{'13':{'median_norm':1}})
        with self.assertRaises(FileNotFoundError):experiment.assert_main_ready()
        experiment.write(self.root/'runs/calibration-provenance.json',dict(tokens_processed=313))
        with self.assertRaises(AssertionError):experiment.assert_main_ready()

    def test_missing_evidence_is_rejected(self):
        with patch.object(sys,'argv',['analyze.py']),self.assertRaises(AssertionError):analyze.main()

    def test_gibberish_probe_is_neither_false_refusal_nor_compliance(self):
        batch=[dict(id='x',compliance_probe=True)]
        rating=dict(id='x',harm=0,coherence=0,behavior='not_applicable',reason='Unintelligible.')
        result=validate_ratings(batch,[rating])[0]
        self.assertEqual(result['behavior'],'not_applicable')
        self.assertEqual(result['scope_note'],'unclassifiable_harmful_probe')
        with self.assertRaises(AssertionError):validate_ratings(batch,[dict(rating,coherence=3)])

    def test_full_fixture_rates_and_blinding_join(self):
        prompts=json.loads((self.root/'configs/prompts.json').read_text());cs=experiment.conditions()
        capdesign=json.loads((self.root/'configs/capability-short.json').read_text());capids={x['id'] for x in capdesign['conditions']}
        raw=[];ratings=[];caps=[]
        for c in cs:
            for p in prompts:
                harm=2 if c['id']=='pair_1' and p['category'] in ['benign','distress'] and int(p['id'].split('_')[-1])<6 else 0
                raw.append(dict(**p,condition=c['id'],response='Fixture response.',token_ids=[1,2,3,4],
                    token_ids_sha256=hashlib.sha256(json.dumps([1,2,3,4]).encode()).hexdigest(),stopped_eos=True,generated_tokens=4))
                ratings.append(dict(id=opaque(c['id'],p['id']),harm=harm,coherence=3,
                    behavior='refusal' if p['compliance_probe'] else 'not_applicable',reason='Synthetic fixture.'))
            if c['id'] not in capids:continue
            for benchmark in ['gsm','mmlu']:
                for i,qid in enumerate(capdesign['questions'][benchmark]):
                    correct=True if c['id']=='baseline' else i%2==0 if c['id']=='pair_1' else i<40
                    caps.append(dict(condition=c['id'],id=qid,benchmark=benchmark,correct=correct,prediction=1,stopped_eos=True))
        for name,rows in [('behavior',raw),('judgments',ratings),('capabilities',caps)]:
            (self.root/'runs'/f'{name}.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
        with patch.object(sys,'argv',['analyze.py']),contextlib.redirect_stdout(io.StringIO()):analyze.main()
        summary=json.loads((self.root/'results/behavior-summary.json').read_text())
        self.assertEqual(summary['baseline']['coherent_harm']['estimate'],0)
        self.assertAlmostEqual(summary['pair_1']['coherent_harm']['estimate'],.2)
        lo,hi=summary['pair_1']['coherent_harm']['ci95'];self.assertLess(lo,.2);self.assertGreater(hi,.2)
        comparison=json.loads((self.root/'results/control-comparisons.json').read_text())
        self.assertEqual(comparison['random_']['exploratory_rank_tail'],1/21)
        cap=json.loads((self.root/'results/capability-summary.json').read_text())
        self.assertEqual(cap['baseline']['gsm']['estimate'],1)
        self.assertEqual(cap['pair_1']['mmlu']['estimate'],.5)
        self.assertEqual(cap['rotated_00']['mmlu']['estimate'],.8)
        self.assertEqual(len(json.loads((self.root/'results/replication-raw.json').read_text())),24)
        self.assertEqual(len(json.loads((self.root/'results/blog-examples.json').read_text())),30)

if __name__=='__main__':unittest.main()
