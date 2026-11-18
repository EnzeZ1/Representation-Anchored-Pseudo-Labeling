# Codex — current request: inspect running experiment results

The active user request is **check existing experiment results now**. Read [CODEX_CHECK_RUNNING_RESULTS.md](./CODEX_CHECK_RUNNING_RESULTS.md) completely. Connect to the existing Campaign A and B machines via the verified `.cs.wisc.edu` SSH hosts and inspect queue state, completed artifacts, per-seed MAE/R2, and paired comparisons. Return actual numerical results when supported.

**READ-ONLY RESULT CHECK ONLY**: the two previously launched training queues must keep running; do NOT rerun, stop, modify, retry, or change any training configurations, tests or checkpoints. Do not perform new held-out-test inference. Do not publish private results to GitHub. The historical RSNA experiment on gpu04 is independent and must not be interrupted.

Prior implementation/launch specification: [CODEX_2080TI_CAMPAIGNS.md](./CODEX_2080TI_CAMPAIGNS.md). The earlier paper-only cleanup instructions in the repository are not part of this task.

Codex's previous job report is only historical evidence; verify all current PIDs, run completion counts and metrics directly on the hosts before asserting a status. A GitHub task file does not itself start a Codex session.
