# Codex task — Recover Q4 clean-dense TE CheX jobs; monitor HPL without interference

## Language / activation
**Use ENGLISH exclusively in the Codex CLI**, including progress updates, final summary and all private reports. Do not reply in Chinese.

This task authorizes the **safe recovery and completion of three already-planned Q4 clean-dense Temporal Ensembling runs**. It also asks for a **read-only status check** of the existing HPL IMDB-WIKI factorial campaign. It does NOT authorize new experiment designs, additional HPL conditions, or restarting already-live HPL jobs.

The GitHub task file is an instruction handoff, not a job launcher. Begin when invoked in the user's active Codex CLI session.

## Prior state to verify, NOT assume current
At the user's latest audited snapshot (Nov 21, 2026, ~20:36 CST):
- Q4 clean-dense TE: **21 of 24 runs complete and valid**; three CheXchoNet jobs interrupted by a machine reboot and have stale RUNNING status. Known unfinished cells: `CheXchoNet / ResNet-50 / seed 2`, `CheXchoNet / DINOv2-S / seed 1`, `CheXchoNet / DINOv2-S / seed 2`. The last observed progress was approximately 29, 26 and 18 fully completed epochs respectively, with partial work in the next epoch.
- Controllers and original job PIDs for these three were absent in that audit.
- HPL factorial: UTKFace **24/24 complete**; IMDB-WIKI **24 core runs reported LIVE** across gpu01, gpu02, gpu04, roughly 7–15 epochs out of 30. These HPL jobs must be monitored **READ ONLY**; do not change their controller, schedule, source, data, or output files.
- MT/TE supervision-strength ablation **36/36 complete** and must not be touched.

First refresh actual host/process/job state (including an alternative agent potentially having restarted these three Q4 jobs). Do not treat stale disk status as real liveness.

## Phase 1 — Q4 inventory and exact protocol
1. Locate the original Q4 24-run manifest, training launcher/controller, source code, full as-run parameters, seed files, labels/manifests, previous recovery log and run output directories on the authorized lab hosts. Work from actual source and run records, not from this summary alone. Verify the exact clean-dense TE variant: native TE targets/temporal ensemble retained, clean supervised term on every prescribed unlabeled step, original objective scales, total steps/epoch, best-validation checkpoint logic, 30 epochs, and intended one-time test evaluation. Do not confuse with TE+Interaction, which adds an extra term instead of replacing native masked supervision.
2. Independently verify all **21 completed Q4 cells** have final metrics, val-selected checkpoint, complete history and correct one-time test guards; do not modify/rerun these cells. Inspect each of the three interrupted cells: full logged epochs, last logged step, source/config hash, best validation snapshot, any in-progress/latest checkpoints, and actual optimizer/RNG/dataloader/ensemble state saved.
3. Determine if an interrupted run can be resumed **faithfully**. For Temporal Ensembling, full resume requires not just neural-network weights and optimizer/scheduler and RNG/data order, but also the **entire temporal ensemble target/accumulator state and any global epoch counters**. Do not use best-validation-only weights to resume training as though nothing was interrupted. If any necessary state is absent or mismatched, label `NOT_EXACTLY_RESUMABLE`.
4. Inspect host uptimes, GPU memory and active processes, disk quota, and concurrent users' jobs. Never preempt shared workloads or service GPUs. Do not stop or change existing HPL trainers.

## Phase 2 — Recover ONLY the three Q4 cells
- If checkpoint state is complete and validated, resume interrupted cell(s) at the exact next step/epoch with the saved deterministic stream. Clearly record true resume vs fresh restart.
- If exact continuation is not technically possible, **restart only that cell from epoch 0**, same source/seed/config/protocol in a **new attempt/output directory**. Leave all old partial artifacts and checkpoints intact. Do not claim those partial epochs were successfully reused. It is better to rerun three scientifically valid experiments than to combine non-matching training states.
- Start/restart jobs only if they are not already live, after obtaining a per-cell idempotent lock or verified unique job ID. Inspect available GPUs at launch time; allocate truly idle GPUs and avoid interference with HPL on gpu01/gpu02/gpu04 or any other users. If nothing is safely available, queue without preemption.
- Use an isolated controller robust to SSH disconnect, with transparent status and crash recovery. Do not change shared system services or source-of-truth experiment manifests. If new instrumentation is needed, confine it to a distinct recovery worktree, validate unchanged scientific training semantics and preserve hashes.
- Ensure val-only checkpoint selection and exactly **one held-out test evaluation** per final run; old partial runs must not already have a final test. No test-based model selection.
- Do not launch a fourth Q4 cell, new experimental conditions, HPL no-BLO P1 extension, HPL repeats, new datasets or any new Q4 campaign.

## Phase 3 — HPL status watch and consolidated reporting
1. **Read only:** identify which of the 24 IMDB-WIKI HPL factorial jobs are actually running, which are completed, interrupted or queued, and check the UTKFace 24 valid cells remain intact. Monitor their log progress and controller health; do **not** relaunch or change them as part of this task. If a crash is found, report it and the safe next action separately rather than making changes without separate authorization.
2. Create new **private local** files (e.g. a distinct recovery/audit folder in `artifacts/`; do not overwrite historic data):
   - `Q4_CHEX_RECOVERY_STATUS.md`: full Q4 24-cell inventory, three-cell diagnosis and handling, live host/GPU/job IDs, currently running vs complete, exact checkpoint resume or epoch-0 restart rationale, file paths and timestamps.
   - `Q4_CHEX_RECOVERY_ATTEMPTS.csv`: each of the three cell IDs, original attempt path, new attempt path, source/config sha, resumed_from, controller/job, status, validation and test evaluation guards.
   - `HPL_IMDB_READONLY_STATUS.md`: IMDB live/completed/interrupted/queued counts and host assignments; note any problems requiring new authorization.
3. For Q4 completed runs, report validation status/finishing count; if not yet completed, accurately report active jobs, never claim done prematurely. Upon all three completing, verify 24/24 complete and update a private combined six-block Q4 clean-dense TE table (dataset × backbone, seeds 0–2, MAE mean±population SD, with paired analyses only where methodologically warranted).
4. No private manuscript, unpublished numerical results, patient information, logs, checkpoints or reviewer identity/results may be pushed to this **PUBLIC** GitHub repository.

## Final CLI reply — ENGLISH ONLY
Give a short, factual status:
- time of live recheck;
- Q4 three-cell diagnosis, resume/restart action actually performed and paths;
- counts `Q4 valid / running / interrupted / queued`;
- HPL IMDB live/completed/interrupted count (read-only);
- any blocker and next action;
- statement that HPL running jobs, original successful Q4 cells and other campaigns were not touched.

No time estimates unless based on the queue scheduler with an explicit source; no speculation about completion.
