# evil-qwen

Phase 1 only: fixed two-neuron Qwen3.5-4B steering, controls, and capability costs.
Heavy work runs in WSL at `~/codeprojects/evil-qwen`, Python `~/evilqwen/.venv/bin/python`; Mac never loads weights. SSH only; preserve PC GUI and power settings.
Prompts, rubric, conditions, and analysis plan in `configs/` are frozen before steered output. Preserve hashes and all failed/truncated rows. Raw outputs in ignored `runs/`; small summaries in `results/`.
Use Codex subscription CLI with clean home for judging; no metered API. No Phase 2 or new neuron searches.
Blocked run: WSL exec failed with `Wsl/Service/0x8007274c`; inspect the task's `evilqwen-corpus` tmux job after access recovers. Preserve unrelated swarm sessions. Final Mac scripts are newer than the guest copy; sync first. Main generation requires source-corpus calibration provenance and matching validation hashes. Pilot checks are not replication results; see `RESULTS.md`.
