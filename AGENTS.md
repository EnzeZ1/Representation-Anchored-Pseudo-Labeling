# Codex — active task on this branch

**Read [CODEX_2080TI_CAMPAIGNS.md](./CODEX_2080TI_CAMPAIGNS.md) completely before editing or launching.** It contains the active, user-approved task: launch 144 UTKFace ratio/frequency/placement runs plus 48 supervised-only controls via SSH to enzez@instgpu-00.wisc.edu through enzez@instgpu-04.wisc.edu, using up to 24 + 12 idle RTX 2080 Ti GPUs and reserving 4 devices if available.

The file `CODEX_PAPER_ONLY_CLEANUP.md` in this repository documents an earlier, separate release task. **Do not perform paper-release cleanup as part of the active 2080Ti task.** The RSNA 48-run experiment on gpu04 is another separate running campaign: do not stop it, overwrite its files or compete with its GPUs.

Create a fresh separate worktree/source path. Perform complete five-host preflight, check actual GPU occupancy and VRAM, then start only scientifically valid workloads. Preserve all existing research code, results, manifests and source profiles. Never claim a job is running without verifying a live PID bound to an idle physical GPU. Never publish private dataset or unpublished result data. Follow the full task spec for exact configuration, validation, logs, and status reports.
