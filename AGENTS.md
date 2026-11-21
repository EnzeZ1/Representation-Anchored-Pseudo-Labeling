# Active Codex task: HPL factorial experiment — launch only after preflight

**LANGUAGE REQUIREMENT: Communicate in ENGLISH in the Codex CLI at all times, including progress reports, analysis and final output. Do not respond in Chinese.**

Read and follow [CODEX_HPL_FACTORIAL_SCHEDULE_VS_BILEVEL.md](./CODEX_HPL_FACTORIAL_SCHEDULE_VS_BILEVEL.md) in FULL. It is the active task instruction for this branch, overriding any other generic/release-agent instructions in older branches. The user authorizes this HPL mechanistic experiment and has many GPUs, but **all new runs are subject to strict method-fidelity and protocol preflight**.

Scientific goal: cross supervision density (native repeat-2 sparse-even vs dense, at exactly matched full U predictor steps) with pseudo-label mechanisms (Uniform vs full HPL, plus a faithful published uncertainty-without-bilevel condition if independently verified). Work on UTKFace and IMDB-WIKI, 5%, R50 and DINOv2-S, seeds 0–2. Exactly 48 core runs, plus 24 conditional published no-BLO ablation runs. Do not presume HPL gains are caused primarily by dense supervision; quantify the factorial.

Use pinned upstream HPL source and previous private history/fairness audit results to prevent official-baseline misattribution. Keep meta updates and meta-sampling routing independent of the sparse/dense predictor schedule. Never use held-out or validation labels for HPL outer learner; use train-labeled data only. Never contaminate the old k-sweep's effective repeat=1 with the new repeat=2 design.

Preflight before committing GPU resources. Reuse only truly idle GPUs and do not affect any existing campaigns or service GPUs. Implement in an isolated new local worktree and distinct run/output namespaces. Run only verified conditions; P1 should be skipped with an explicit fidelity reason if the official no-BLO learner training rule cannot be established. No new data sets, no six-seed expansion without authorization.

**PUBLIC GITHUB RESTRICTION:** This is a public repo. Do not push private numerical results, rebuttal text, reviewer data, checkpoint files, raw data, personal identifiers or credentials. Keep experiment artifacts and reports on the authorized private host. Merely fetching this branch never starts the work; it must be executed in the user's active Codex CLI session.
