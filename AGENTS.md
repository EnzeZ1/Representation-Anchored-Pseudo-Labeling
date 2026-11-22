# Active Codex Task: Q4 Clean-Dense TE CheX Recovery + Read-Only HPL Watch

**LANGUAGE: All CLI communication, status updates, analysis, private reports, and final summaries MUST be in English. Do not respond in Chinese.**

Read [CODEX_Q4_CHEX_RECOVERY_HPL_WATCH.md](./CODEX_Q4_CHEX_RECOVERY_HPL_WATCH.md) **in full**. It is the active task instruction for this branch, overriding any other default/release instructions.

Scope: **recover exactly three previously interrupted Q4 clean-dense Temporal Ensembling CheXchoNet experiment cells** (ResNet-50 seed 2, DINOv2-S seeds 1 and 2) after validating current job state. The user authorizes safe resumption if and only if full training state is recoverable (including TE ensemble targets); otherwise rerun from epoch 0 in new isolated attempts, preserving original incomplete artifacts. Do not rerun valid completed Q4 cells.

**HPL factorial IMDB-WIKI monitoring is READ ONLY.** These 24 already-authorized jobs were reported live after previous reboot recovery. Do not stop, restart, relaunch or modify them. Do not schedule new HPL conditions, including no-BLO P1.

Verify actual GPUs/PIDs/controller states before action and protect other users' and other campaigns' jobs. Keep all raw results and reports private on lab hosts; this public GitHub branch holds instructions only. A GitHub commit does not itself execute Codex.
