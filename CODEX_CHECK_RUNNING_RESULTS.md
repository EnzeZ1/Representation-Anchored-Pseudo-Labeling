# Codex — Check live results for the two 2080 Ti rebuttal campaigns

## Immediate instruction

The user asks: **"Check the results now."** This is a **read-only status/results audit**, not another launch request. Do not start additional experiments, change hyperparameters, modify/requeue/stop any live jobs, edit source, consume reserved GPUs, run new test inference, or overwrite any artifacts. Existing campaign controllers must keep running undisturbed.

When invoked in the user's Codex session, connect to the hosts and **actually inspect the on-disk logs, queue state and results**. Do not treat the last Codex report as live status. Use only the existing authorized SSH credentials, never request or reveal secrets.

Host aliases already verified by the previous Codex session:
```bash
ssh enzez@instgpu-01.cs.wisc.edu  # Campaign A (144)
ssh enzez@instgpu-02.cs.wisc.edu  # Campaign B (48)
```
The original `.wisc.edu` short suffix was not resolvable. gpu04 has a separate running RSNA 48-run project; you may **read** its status opportunistically if accessible, but it is **not** part of the 192-run counts and you must not affect it. Other hosts are not needed for these two campaigns unless their verified queues were subsequently moved.

Expected worktree on **both** hosts:
`/nobackup/enzez/ssl_2080ti_worktree_20261117`

Expected output root:
`artifacts/ssl_controls_2080ti_20261117/`

Prior controller names:
- gpu01: tmux `ssl2080-A-20261117`, controller PID previously 1012603, GPUs 0–7, campaign A.
- gpu02: tmux `ssl2080-B-20261117`, controller PID previously 580962, GPUs 0–3, campaign B.

Previous status at launch, **not the current status**: A 144 eligible / 0 complete / 8 running / 136 queued; B 48 eligible / 1 complete / 4 running / 43 queued. No blocked/failed at that time. These numbers must be recomputed, not copied.

## Steps

1. Connect to both hosts, determine present time and source tree revision, inspect `tmux ls`, `ps`, `nvidia-smi`, controller log, queue JSON/CSV and the last updates in `QUEUE_STATUS.md` or other status files. Verify whether each controller is alive, which workers are actually running and whether queue state is stale. **Do not use SIGINT, tmux send-keys, scheduler restart, or any GPU-consuming operation.**
2. Count all **144** distinct campaign A keys and all **48** distinct campaign B keys. Report statuses completed / active-running / pending / failed / blocked separately. Flag missing IDs, duplicates, stale PIDs, unexpected output paths and queue stalls. Note any unavailable host or broken SSH and do not infer its current state from old logs.
3. On each completed run, verify persisted `config.json`, `metadata.json`, full epoch history (30 epochs when expected), best validation-selected checkpoint, `metrics.json`, outcome metric, seed/ratio/backbone/condition, source and manifest hashes, and evidence of a single final held-out-test evaluation after checkpoint reload. **Read existing result files only**. Do not call test inference to reconstruct missing metrics and do not reinterpret partial validation scores as final test scores. Mark suspicious/incomplete as `invalid_or_unverified`, not successful.
4. Where complete and valid, extract **test MAE and R2** in original units, and run basic numeric checks (finite, plausible units, no duplicates, correct split). Summarize the actual per-condition values, not only controller status. Preserve the distinction between test MAE (years for UTKFace/IMDB-WIKI, months for RSNA, cm for CheXchoNet) and normalized train loss.
5. Campaign A: for each `ratio ∈ {0.01,0.10,0.20}` and backbone, compare available `front/even/back` at **the same paired seed** with identical label budget `L`, unlabeled steps `U`, source protocol, test split and selected model protocol. Likewise summarize `k ∈ {1,2,4,8,20}` frequency sweep at each ratio/backbone. Report test MAE by seed, aggregated mean/SD only when appropriate, and the **number of complete paired seeds**. Only calculate a paired 95% CI for actual common seeds, clearly flag `n=1/2/3` and its uncertainty (with n<2, no SD/CI). A 3-seed t-interval is exploratory; do not call it conclusive. Don't assume frequency controls labeled exposure—only placement does.
6. Campaign B: report `sup_only_one_pass` versus `sup_only_recycled` paired seed comparisons for each dataset and backbone. Carefully distinguish their optimizer-step counts and labeled exposure. If corresponding previous **Plain+Interaction** data with the **same manifests, preprocessing, backbone, seed, optimizer and selection rule** is available locally, add a separately labeled matched comparison; otherwise state that this comparison is not yet justified rather than mixing experiments. Never compare incompatible historical artifacts.
7. Optional: check the **status only** of the separate gpu04 RSNA 48-run `front/even/back/dense_mass_matched` study if an existing read-only, verified status file is available, without touching its jobs. Its results must be tabulated separately and excluded from A/B statistics.
8. Inspect error traces for failed items and provide likely failure classes (OOM, file access, bad manifest, CUDA/runtime, stall, invalid metric). **Do not restart, retry, relaunch or patch anything** in this check task. Mention precise recommended remedy as a suggestion only.
9. Write a fresh **timestamped local** read-only audit report, e.g. `artifacts/ssl_controls_2080ti_20261117/RESULT_CHECK_20261118.md` on the respective host or in a neutral reporting directory, using atomic creation. The report may include internal run IDs and summary statistics but **must not be pushed to public GitHub**. If report output may interfere with the active queue or paths, write outside the queue folders. Include commands to monitor progress and the next point at which a meaningful conclusion can be drawn.

## What to return to the user immediately

**Don't answer only "running" or give only a file path.** In the Codex final report, present:
- Fresh audit timestamp and whether both SSH sessions worked;
- For **A and B separately**: complete/running/queued/failed/blocked/invalid; verify totals 144 and 48;
- Which conditions now have sufficient completed paired seeds for a preliminary MAE/R2 comparison;
- **Actual early results** (mean/SD or seed-level MAEs) for eligible matched groups, with explicit pairing status, or say exactly why no valid comparisons are yet available;
- Any unexpected trends that could matter for x7tN/AfNK reviewer response, explicitly marked exploratory until complete;
- Existing errors, stale jobs, bottlenecks, and any action the user should take;
- Exact local paths to the detailed audit reports and selected log/metrics files.

**Protect scientific integrity.** No assumptions based on the paper table, no fabrication of completed runs, no numerical claims based on partial epochs, no changing the experiment or cherry-picking reported seeds. Keep results private, preserve live RSNA/2080Ti training, and avoid new test inference.
