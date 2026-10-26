# evil-qwen

Phase 1 only: fixed two-neuron Qwen3.5-4B steering, controls, and capability costs.
Heavy work runs in WSL at `~/codeprojects/evil-qwen`, Python `~/evilqwen/.venv/bin/python`; Mac never loads weights. SSH only; preserve PC GUI and power settings.
Prompts, rubric, conditions, and analysis plan in `configs/` are frozen before steered output. Preserve hashes and all failed/truncated rows. Raw outputs in ignored `runs/`; small summaries in `results/`.
Use Codex subscription CLI with clean home for judging; no metered API. No Phase 2 or new neuron searches.
Initial WSL stall was recovered by the lead with `wsl --shutdown`. Resume uses fixed local shards, tokenizes before loading weights, and checkpoints calibration norms. Run long jobs through `supervise.py` in tmux: 30s heartbeats, Windows-readable status, aggregate RSS/available-RAM guards. Main generation requires corpus calibration provenance and matching validation hashes. Pilot checks are not replication results.
Generation must explicitly stop at `<|im_end|>` (248046), plus configured EOS248044; default EOS alone continues into fabricated turns. The completion-only pilot is archived and excluded. Capability amendment: first50 of each seeded100 sample, 13 conditions, 25min; use `capabilities_short.py`. Opener counts are literal post-hoc styles, not semantic compliance.
