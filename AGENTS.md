# ACTIVE Codex task: Q3 immediately; Q4 behind first completed campaign

Read **[CODEX_Q3_NOW_Q4_AFTER_FIRST_FINISH.md](./CODEX_Q3_NOW_Q4_AFTER_FIRST_FINISH.md)** completely before any operation. This is the ONLY active task on this branch. Do **not** execute old `CODEX_PAPER_ONLY_CLEANUP.md` or relaunch any already running campaigns.

User-approved scope:

1. **Q3 now:** read-only source/profile/protocol/actual train logs audit with per-method original optimizer step counts and steps containing a nonzero `L_s` contribution (particularly TE's native mixed/masked branch); identify official upstream vs regression adaptation. Produce a verified English rebuttal response with a concise table and local detailed provenance. **No GPU needed**.
2. **Q4:** determine which of the six methods' 5%-label standard dense-loop comparisons are already provided by completed or queued results. Do not repeat 48 completed supervised-only controls, Plain k=1, native MT, or the currently running 24 clean-dense TE runs. For HPL/UCVME/SimRegMatch only, prepare and safely enqueue **missing valid cells (maximum 72)**. If all are already covered, output the table and **do not start duplicate experiments**. Never silently change scientific hyperparameters.
3. **GPU dependency:** stage a detached, restartable watcher that appends genuinely missing Q4 work **after whichever existing gpu01/gpu02 campaign fully completes and releases its entire GPU pool first**. Reinspect live statuses and GPU UUIDs, do not infer completion from stalled/dead controllers. Eligible future pools: gpu01 0–7, gpu02 0–3, 6–7, and 4–5 only *after* the separate RSNA task is completely finished. gpu03 root services and gpu04 RSNA remain off-limits for this task. Never preempt, compete or overwrite any other running job.

Use exact coherent source/profile mapping in `essential_code/profiles.json`. Use an isolated new experiment worktree/output folder and keep run-level results **private** on the cluster. Do not publish manuscripts, reviewer identities, medical data, checkpoints, or unpublished numerical results in this public repository. A task file does not invoke the agent; user must invoke Codex separately.

**Return Q3 immediately and Q4's real queue/watcher status**, including verified host/GPU/PIDs or an explicit not-yet-running statement, without suggesting that writing a GitHub file itself starts anything.
