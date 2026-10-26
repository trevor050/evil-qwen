# Resume execution changes, 2026-10-26

Lead recovered the stalled WSL VM with `wsl --shutdown`; the executor did not perform that shutdown. Resume retains frozen prompts, rubric, neuron indices, controls, decoding and analysis design.

Replace all-shard HF streaming calibration with direct downloads of one pinned FineWeb-Edu sample-10BT shard (`013_00000.parquet`) and one pinned Ultrachat train_sft shard. Files are processed locally, in bounded Arrow record batches, with a seeded shuffle buffer of 128 examples. Alternate documents with EOS separators and tokenize before loading weights. This is an independent shard-restricted calibration, not the author's original all-shard sample. Hash the shards and actual token stream.

Calibration inference uses 1024-token windows, batch2, layers through 14 only. Save atomic norm chunks every ~32k tokens and estimates at the first batches exceeding 200k, 500k, and 2M tokens. Resume at the last complete chunk. Target2M remains primary; the lead permits 500k only if 2M cannot finish reasonably. The checkpoint mean/median curve will show actual residual-norm convergence; source observations about corpus maxima do not establish median convergence.

Every long stage runs inside task-specific tmux under `scripts/supervise.py`. Heartbeats every30 seconds include elapsed time, aggregate child RSS, Linux `free -m`, and latest work progress. A small status file also lives under Windows `%LOCALAPPDATA%/Temp/evilqwen-progress/` so Windows SSH can read the last heartbeat if guest exec fails. Stop only the job's process group above9GiB aggregate RSS or below2200MiB available system RAM. Release this experiment's checkpoint/shard file cache with `posix_fadvise`, without changing global VM settings.

Blind judging can run incrementally on completed rows while the GPU continues generating. Each pass independently shuffles its snapshot, uses opaque IDs and ephemeral clean-home CLI calls, and records ratings by immutable (condition,prompt) hash; the final pass resumes unjudged IDs. Prompt text, rubric, thresholds, and generation settings remain frozen. No condition labels enter judge input.

## Assistant-turn termination repair (before final analysis)

The raw-token audit found the checkpoint generation config stopped on `<|endoftext|>` but not `<|im_end|>`. The initial completion run could therefore generate fabricated subsequent user/assistant turns. The actual explorer runtime explicitly stops at `tok.convert_tokens_to_ids("<|im_end|>")` (runtime.py line181 at the pinned commit). This is a harness defect, not an experimental arm selected for stronger effects.

Preserve all completion-run rows, ratings, logs, source script and partial tables under `runs/completion-eos-pilot/`; exclude them from primary metrics. Restart validation and every behavioral condition with generation explicitly stopping on the first assistant end token (or configured EOS). Calibration, prompts, rubric, conditions, doses, output cap, batch size and benchmark sample stay unchanged. Preliminary completion-run scores quoted during execution are superseded. The screenshot's suicide-encouragement sentence was already inside the first assistant turn, so its literal presence was not caused by the extra turns; all final rates nevertheless use fresh corrected generation and blind rating.
