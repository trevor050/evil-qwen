# Qwen3.5 two-neuron steering, Phase 1

> **This is lyra's discovery, not mine.** [lyra](https://x.com/_lyraaaa_/status/2113833330429083662) found the two neurons and the recipe, and built [neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer) to do it. All I did was measure it carefully. If this is interesting to you, go show her some love.
>
> **What this is and is not.** In lyra's words, this is a personality tune, not a refusal abliteration or a decensor: the steered model acts like an over-the-top villain and gets much worse at nearly everything else. It has no practical use. It is here because two neurons changing a model's whole demeanor is worth measuring carefully, with frozen prompts, blind scoring and controls.
>
> More research from the same lab: [Honorly Research](https://research.honorlyai.com).

A replication and controlled evaluation of the supplied L14 N8949 +1 / L13 N7956 -1 recipe. The source is [lyramakesmusic/neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer/tree/f79b2196e29aacbad89684938baac8d97b172fd1), not the unrelated TinyStories neuralese site. Results and limitations are recorded in `RESULTS.md`; the initial blocked handoff is preserved in Git history.

All model work runs in WSL on the RTX 4070. Use SSH only. Keep the desktop, power settings, and other workloads intact. Mac code only transports files and edits reports.

## Environment

WSL checkout `~/codeprojects/evil-qwen`; isolated environment `~/evilqwen/.venv`.

```sh
uv venv ~/evilqwen/.venv
uv pip install --python ~/evilqwen/.venv/bin/python -r requirements-lock.txt
```

The model loaded successfully with torch 2.14.1+cu130 and Transformers 5.20.2. `requirements.in` records direct dependencies; `requirements-lock.txt` records the installed environment. Downloaded model snapshot is fixed in `scripts/run.py`. Model uses the supported text-only `Qwen3_5ForCausalLM` loader, which remaps multimodal checkpoint keys and omits vision/MTP weights. Inference uses bf16 and SDPA with FLA chunk/recurrent kernels; convolution uses the HF PyTorch fallback. No vLLM, quantization, compilation of the full model, API keys, or paid inference.

From Mac, synchronize code with `python3 scripts/remote.py sync`. Remote commands can be wrapped as `python3 scripts/remote.py exec /home/user/evilqwen/.venv/bin/python scripts/run.py STAGE ...`. Long jobs run in task-specific tmux sessions. Do not use the Mac Python for model stages.

## Reproduction commands, in WSL checkout

Set `PY=~/evilqwen/.venv/bin/python`, then run one command per stage:

```sh
$PY scripts/supervise.py download $PY scripts/prepare_corpus.py download
$PY scripts/supervise.py tokenize $PY scripts/prepare_corpus.py tokenize
$PY scripts/supervise.py cal200 $PY scripts/run.py calibrate-corpus --batch 2 --tokens 200000
$PY scripts/supervise.py cal2m $PY scripts/run.py calibrate-corpus --batch 2 --tokens 2000000
$PY scripts/supervise.py validate $PY scripts/run.py validate
$PY scripts/supervise.py behavior $PY scripts/run.py behavior --batch 8
$PY scripts/supervise.py judge $PY scripts/judge.py --workers 12 --batch 12
$PY scripts/run.py prepare-capabilities
$PY scripts/supervise.py metadata $PY scripts/capture_metadata.py
$PY scripts/supervise.py capabilities $PY scripts/capabilities_short.py --batch 8
$PY scripts/capture_resources.py
$PY scripts/analyze.py
$PY scripts/write_results.py
```

Behavior and capability stages resume by completed (condition,prompt) keys, retaining raw text and token IDs. Changing scripts/configs/model mid-run invalidates comparisons; preserve run manifests. Calibration must finish and validation must pass before generation. Main stages reject ordinary-prompt pilot calibration and stale validation hashes. Dataset sampling uses fixed seed 20261025; source calibration uses source seed0. Bootstrap seed and resample count are frozen in `configs/analysis-plan.md`.

Judging uses only the Codex subscription CLI `~/.local/lib/nodejs/bin/codex`, model `gpt-6-luna`, medium effort, an isolated `CODEX_HOME=~/evilqwen/codex-home`, empty cwd, read-only sandbox, and a strict JSON schema. `auth.json` is a local symlink to `/mnt/c/Users/Trevor/.codex/auth.json`; it is never copied into the repo. API-key overrides are removed. Judge inputs contain opaque IDs, prompts, responses, and the compliance-probe flag, with no condition names. Invocation logs, retries, and mapping stay under ignored `runs/`. `--smoke` tests the CLI with a trivial arithmetic fixture.

## Frozen design and evidence

Initial design/rubric/prompts were committed before steered behavioral outputs. Lead corrections are preserved as ordered, hashed amendments: `indexing-amendment.md` (later superseded), then `source-amendment.md`. The original fixed prompt files and rubric remain byte-identical. `freeze_design.py` is provenance, not a routine command to rerun after observing results.

There are 51 required behavioral conditions: unhooked baseline, dose-zero hooks, six nonzero paired doses, each neuron alone, flipped signs, 20 random-neuron pairs, and 20 rotated-direction pairs. The shifted-index arm is deferred unless literal zero-based steering fails to reproduce. Random controls use both original source layers and the same residual-norm dose.

Calibration follows the source measurement in `neuron_stats.py`, adapted to fixed local shards after a WSL hang: alternate locally shuffled FineWeb-Edu and Ultrachat documents, append EOS, pack 1024-token windows, retain the first token in inference but exclude it from norm statistics, measure post-residual decoder-layer outputs before subsequent/final RMSNorm. See `docs/resume-execution.md` for shard selection, bounded shuffle, checkpoints, and resource guards. The original ordinary-prompt calibration is retained under `runs/pilot-ordinary-calibration/`, separated from main results. Source fallback calibration and the original author's actual scalar values may differ; independent calibration is not an exact reconstruction of their corpus.

The lead shortened capabilities before results: first50 questions from each seeded100 sample; baseline, pair_1, only_L14 and first5 controls per family, 13 conditions total, with a25-minute stage budget. `capabilities_short.py` waits for full behavioral judging, writes a pending-capability behavior report, then loads the GPU as the final inference stage. See `docs/capability-amendment.md`. The exploratory opener tables and30 verbatim blog examples are generated by the analysis stage.

For MMLU we evaluate the argmax first-token logit among A/B/C/D under the non-thinking chat template. This is constrained greedy multiple-choice classification, not free-text MMLU or few-shot official leaderboard scoring. GSM8K uses greedy free generation capped at 192 tokens and a documented numeric parser. Truncations and parser failures are reported.

Full generations, activation-norm arrays, calibration token IDs, and judge logs stay in ignored `runs/`; aggregate CSV/JSON and 12 paired raw examples are committed in `results/`. Downloaded Parquet shards are excluded from artifact transfers; their pinned revisions and hashes are retained. Fetch the experiment artifacts to Mac with `python3 scripts/remote.py fetch`. `results/artifact-sha256.json` records raw artifact hashes. Never commit authentication files or raw credentials.
