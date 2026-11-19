# Repository Agent Instructions — GPU rebuttal experiment handoff (2026-11-18)

**ACTIVE TASK: user-approved W2/Q4 experiment deployment.** On this branch read the following THREE documents completely before doing anything:

1. [CODEX_W2_MATCHED_36_RUNS.md](./CODEX_W2_MATCHED_36_RUNS.md): gpu01 GPUs 0–7, **36** fresh paired supervised-only-recycled vs dense Uniform experiments for IMDB-WIKI, RSNA, CheXchoNet 5%.
2. [CODEX_Q4_CLEAN_DENSE_TE_24.md](./CODEX_Q4_CLEAN_DENSE_TE_24.md): gpu02 GPUs 0–3, audit existing native/dense source/experiments and **24** clean standard dense Temporal Ensembling runs on 4 datasets at 5%.
3. [CODEX_MT_TE_SUPERVISION_ABLATION_36.md](./CODEX_MT_TE_SUPERVISION_ABLATION_36.md): gpu02 GPUs 6–7, **36** controlled native vs doubled-native supervised coefficient vs extra-recycled labeled batch runs (Mean Teacher/Temporal Ensembling on UTKFace 5%).

**Do not launch anything just from this file's existence.** These instructions apply only when the user invokes Codex. If executing now in the user's authorized Codex session, run preflight and launch all three approved queues on their **disjoint** GPUs, verifying real PIDs/logs, with noninterfering separate source worktrees, output roots, tmux sessions and controller locks. Coordinate use of gpu02 shared disk and cluster IO. No cross-task accidental duplicate runs or source modification.

Hardware based on the latest user report, not yet live-verified at invocation: gpu01 0–7 idle, gpu02 0–3 and 6–7 idle, gpu02 4–5 occupied with **ongoing RSNA 10/20% experiments**, gpu03 occupied by root services, gpu04 occupied by earlier RSNA experiments. The nonowned or already assigned GPUs **must never be borrowed**, even if 0% utilization. Do not kill/preempt/overwrite any other run. Recheck GPU UUID, process list, VRAM, host, and queue ownership live before starting.

Treat `essential_code/profiles.json` as the authoritative archive mapping. `essential_code/shared` is **not** a single coherent execution profile. Preserve all historical files, dataset manifests, checkpoint and prior results. Separate immutable experiment code, deterministic run matrices, fresh isolated outputs, validation-only checkpoint choice, one postselection test pass, full per-seed provenance and strict source/config equality audits. Never silently modify source or training hyperparameters to get an experiment to fit hardware.

The repository is public. Keep manuscript, reviewer identities, unpublished results, medical data, credentials, private logs and experiment outputs off public GitHub; public branch contains *instructions only*. Report results directly to the user and store complete reports server-locally.

For each campaign provide timestamped verified host+GPU mapping, number completed/running/queued/failed/blocked, controller and trainer PIDs, log root, commands to monitor and exact scientific limitations; no fabricated launches. An agent task file is **not** itself a launch.
