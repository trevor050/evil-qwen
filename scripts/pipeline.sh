#!/usr/bin/env bash
# Run in WSL after source calibration; every long stage has a separate supervisor.
set -euo pipefail
cd /home/user/codeprojects/evil-qwen
PY=/home/user/evilqwen/.venv/bin/python
LOG=/home/user/evilqwen/logs
mkdir -p "$LOG"
"$PY" scripts/supervise.py validate "$PY" scripts/run.py validate > "$LOG/validate.log" 2>&1
"$PY" scripts/supervise.py behavior "$PY" scripts/run.py behavior --batch 8 > "$LOG/behavior.log" 2>&1
"$PY" scripts/supervise.py judge "$PY" scripts/judge.py --workers 12 --batch 12 > "$LOG/judge.log" 2>&1 &
JUDGE_PID=$!
"$PY" scripts/supervise.py capabilities "$PY" scripts/run.py capabilities --batch 8 > "$LOG/capabilities.log" 2>&1
wait "$JUDGE_PID"
"$PY" scripts/analyze.py > "$LOG/analysis.log" 2>&1
date -u > "$LOG/pipeline-complete.txt"
