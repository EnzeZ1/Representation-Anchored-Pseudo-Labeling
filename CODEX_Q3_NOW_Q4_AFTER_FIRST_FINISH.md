# Codex execution task — Finish Q3 now; append only missing Q4 dense-loop controls after the first finished queue

## Scope and authorization (2026-11-19)

The user explicitly requests:
1. **Immediately finish Q3 by source/config audit without consuming a GPU.** Provide a verified, concise, paste-ready table about native supervision frequency and official-code provenance.
2. **Prepare and actually enqueue the missing Q4 FixMatch-style dense-loop controls behind whichever of the currently running experiment campaigns completes soonest.** Do NOT preempt, interrupt, modify, restart, or compete with active experiments. The queue must start automatically **only after** the chosen campaign has COMPLETELY finished, its controller is no longer dispatching jobs, and the selected GPU IDs are actually free. Q4 requires an exact no-duplicate audit BEFORE deciding any new cells.

This is a Codex handoff, not evidence any remote activity has occurred. Once invoked by the user in their live Codex session, connect, perform the Q3 audit and prepare the Q4 task/queue; return verified actual statuses. Do not merely print an experimental suggestion.

**Private scientific details:** the repository is public; manuscript, unpublished numeric results, reviewer identities, sensitive/protected datasets, raw logs, credentials and checkpoints must stay on private hosts. Do not publish these documents or a pull request containing detailed result numbers. Use public GitHub instruction files only.

## Prior campaigns and protected resources

Last user-reported campaign status is a *historical* snapshot; ALWAYS inspect live state. Known existing work:

| Existing campaign | Host: GPU IDs | Controller / tmux | Artifact root |
| --- | --- | --- | --- |
| 36-run matched supervised-only recycled vs dense Uniform consistency (W2) | gpu01: 0–7 | PID 160745; w2-matched36-20261118 | /noback/enzez/w2_matched_36_worktree_20261118/artifacts/w2_matched_36_20261118/ |
| 24-run clean-dense Temporal Ensembling (Q4) | gpu02: 0–3 | PID 1736040; q4-clean-te24-20261118 | /nobackup/enzez/q4_clean_dense_te_20261118/artifacts/q4_clean_dense_te_20261118/ |
| 36-run Mean Teacher / TE native vs boosted-mass vs recycled extra (ablation) | gpu02: 6–7 | PID 1736216; mt-te-supervision36-20261118 | /nobackup/enzez/mt_te_weight_ablation_20261118/artifacts/mt_te_weight_ablation_20261118/ |
| 48-run RSNA 10%/20% schedule extension | gpu02: 4–5 | prior controller in 84-run campaign, inspect | /nobackup/enzez/rebuttal_84_worktree_20261118/artifacts/rebuttal_extensions_20261118/ |
| earlier RSNA 5% schedule campaign | gpu04: 0–7 | separate active campaign; READ ONLY if needed | separate original RSNA worktree |

**Important:** the gpu01 W2 path above contains a spelling hazard; the verified report path was:
`/nobackup/enzez/w2_matched_36_worktree_20261118/artifacts/w2_matched_36_20261118/`
Use the **/nobackup** path and verify directory exists. Paths, PIDs and GPU occupancy may change after this document is written.

SSH:
- `ssh enzez@instgpu-01.cs.wisc.edu`
- `ssh enzez@instgpu-02.cs.wisc.edu`

**Eligible Q4 GPU pools after THEIR OWN full campaign exits:**
A) gpu01 0–7; B) gpu02 0–3; C) gpu02 6–7; D) gpu02 4–5 after the RSNA extension ends. Do not use gpu03 (root-owned services) or gpu04 in this task. Never acquire a GPU while its old controller could still dispatch work. Controller exit alone is insufficient; verify all job keys, ps/nvidia-smi, VRAM and no orphan workers. If old campaign is incomplete, stalled, or failed, do not treat it as finished to grab GPUs.

## Part 1 — Q3: finish immediately (CPU / read-only audit; no GPU wait)

Question to answer: For **every baseline's actual paper Original training loop**, how many optimizer steps per epoch include a supervised term `L_s`? Which native loops are faithful to official upstream code (especially TE)? Is the source modified by the paper's regression adaptation? Distinguish a supervised objective *computed* in each step from a **nonzero effective supervised gradient**.

Use **exact locally executed, as-run configs and source hashes**, not the first historical file encountered. Map source variants using `essential_code/profiles.json` and `essential_code/SOURCE_INDEX.csv`. Resolve dataset/seed/backbone-specific steps from per-run `config.json`, loader metadata, scripts, trainers, manifests and, if required, a CPU-only replay of loader index lists; never infer all counts from the UTKFace 5% Plain 29/563 numbers.

Required methods and known implementation pitfalls:
- **Plain/Uniform:** UTKFace 5% original has **29 supervised-containing steps among 563 optimizer steps**. One labeled pass then 534 unlabeled-only steps. For other datasets compute their own L,U.
- **Temporal Ensembling:** Our regression port derives from official `s-laine/tempens` (`a32743acbdea54560b7d9d90934c1eb5b49a81a7` in historical source). Native uses a shuffled **mixed labeled+unlabeled dataset** and `sup=((pred-y)**2 * mask).mean()`. The optimizer objective evaluates the masked term at every mixed step, but an all-unlabeled batch gives zero supervised gradient; count the actual **nonzero-supervision batch steps** and total native steps by seed/dataset/loader. If the precise shuffled ordering is irrecoverable from recorded metadata, state an exact range or unavailable; do not invent a count. The +Interaction version adds an independent coefficient-1 recycled full labeled MSE; it is NOT merely making native masked loss dense.
- **Mean Teacher:** native TwoStreamBatchSampler, at least one labeled example every native step, labels recycled by sampler. Official origin `CuriousAI/mean-teacher` commit `546348ff863c998c26be4339021425df973b4a36` (UTKFace regression adapter). +Interaction adds a second supervised MSE, not an initially missing label term.
- **HPL:** audit native `ssl_mult` labeled dataset-level duplication, predictor step loop, labeled iterator recycling (or lack of it), uncertainty/meta learner steps separately. Official upstream + local adaptation; distinguish if official HPL already has frequent labeled supervision. Original predictor supervised at EVERY executed native predictor step, but number of steps may be limited by `len(labeled_loader)` rather than U. +Interaction may extend epoch updates to U, changing compute as well as label recycling; don't call extra steps solely an original supervision-starvation fix.
- **UCVME:** native optimizer loop controlled by labeled-loader length, labeled L_s every executed step; +Interaction extends to U with labeled cycling. Count model optimizers properly without double-counting twin network updates.
- **SimRegMatch:** native optimizer loop likewise controlled by labeled samples, supervised L_s each executed native step; +Interaction extends to U. Record existing `np.percentile(...,0.95)` behavior as a separate known issue, but do NOT silently correct it in an audit of as-run experimental provenance.

For each method and ideally each 5% dataset/backbone/seed, produce:
`Method | Original optimizer steps/epoch | Original effective L_s steps/epoch | Native labeled loader recycling? | Official upstream repo@commit / local changes | +Interaction change`.
Include numeric per-dataset/seed rows in a private appendix and a compact main table with `L`, `U`, mixed-step notation when numbers differ by seed. Explicitly distinguish **official upstream's native algorithm** from **our regression benchmark adapter**: adapted code is not a verbatim official run. Mention TE calculation's batch-size normalization (`mask` then `.mean()`) and missing-gradient batches.

Write `Q3_NATIVE_LOOP_VERIFIED.md`, `Q3_NATIVE_LOOP_COUNTS.csv`, and a **short complete English rebuttal response** `Q3_RESPONSE.md` locally. Validate every numeric count against exact as-run config or computed loader. Report result to user in Codex final output **now**; no GPU jobs or wait required. If provenance is unavailable, mark the specific cell unresolved and why; never falsely claim fully verified.

## Part 2 — Q4: audit before launching; only scientifically missing cells

Question: supervised-only one-pass/recycled baselines on all 4 datasets × 2 backbones, and **the original six SSR objectives in a standard FixMatch-style dense supervised/unlabeled loop**.

**Already-complete / already-scheduled:**
- 48/48 completed supervised-only 5% controls across 4 datasets × 2 backbones × seeds 0,1,2; DO NOT rerun. Additional UTKFace 1/10/20% supervised-only 36/36 also complete; no need to repeat.
- Plain+Interaction k=1: existing dense single labeled `L_s` per unlabeled step where run provenance validates; do not rerun when verified.
- Mean Teacher native: its TwoStreamBatchSampler gives labeled data every native step; classify that as its native dense counterpart, rather than erroneously treating MT+Interaction's **additional** labeled MSE as the standard loop.
- Temporal Ensembling: **the existing 24-run clean-dense TE queue** across all 4 datasets/backbones × seeds 0,1,2. UTKFace 6/6 already complete at last report. Do NOT duplicate the 24 existing keys. Clean-dense loss uses one full recycled supervised MSE at each native mixed step, replacing native masked-supervised loss in optimizer objective, while retaining the original TE temporal-consistency target and mixed-batch mechanism.
- HPL, UCVME, SimRegMatch: **first** assess exact previous +Interaction source and completed paper runs. If each genuinely uses exactly one coefficient-1 labeled supervision term at every designated unlabeled/native step with otherwise intact method objective, mark as existing **dense loop** (with clear step-budget differences vs native). HPL may have official labeled duplication/cycling already. If the published result is missing, has mismatched manifests, or isn't truly the standard dense interpretation, mark the precise `method,dataset,backbone,seed` key missing.

Create `Q4_EXISTING_VS_MISSING_MATRIX.csv` with all 6 × 4 × 2 × 3 = **144** 5%-ratio method/dataset/backbone/seed dense cells. Columns: method, dataset, backbone, seed, native_source, native_supervised_updates, native_optimizer_steps, dense_existing_valid, dense_existing_result_path, dense_variant_semantics, dense_queued_elsewhere, missing_reason, needs_new_run. A completed paper run may have six seeds: seeds 0–2 suffice for this specifically bounded gap-filling campaign; do not cherry-pick results or select seeds opportunistically.

**New training authorization: ONLY the missing HPL/UCVME/SimRegMatch dense cells** (up to 3 methods × 4 datasets × 2 backbones × 3 seeds = **72** runs). For TE, MT or Plain, an unresolved ambiguity triggers reporting/triage, **not an automatic second run**. If the audit establishes that all 72 are already represented by valid +Interaction results, **no new GPU runs are needed**; generate the Q4 comparison table and tell the user rather than manufacturing duplicate experiments to use GPUs. No additional algorithmic comparisons, label ratios, tuning experiments, or 6-seed expansions.

For genuinely missing cells:
- One designated unlabeled/full native step each iteration, and exactly one coefficient-1 labeled batch MSE term at every such step; recycle labeled batches only as needed. Preserve the full original method-specific pseudo-label/uncertainty/teacher behavior and any native auxiliary/meta optimizer cadence. Do not add a second supervised term to already supervised-containing steps; if native loop must be extended to U, record added optimizer steps and source differences and do not call this fixed-compute unless actually true.
- Follow coherent historical per-dataset/backbone protocol (batch size, drop_last, optimizer/groups, scheduler, pretrained model, data transforms, RNG, label scaler, checkpoint selection). Ensure consistency with the specific method's baseline rather than blindly imposing UTKFace's or RSNA's recipe. Never silently change batch size or precision on OOM. Dataset and benchmark provenance verified from local as-run configs; frozen `essential_code/shared` is **not** a coherent runnable snapshot without profile mapping.
- Three predeclared seeds 0,1,2 at 5%, 30 epochs, validation-only best model selection and one held-out test inference after reload, config/source/manifest/initial state SHA256 + full output records.
- Ensure the adapter actually runs in FP32 on RTX 2080 Ti with unchanged scientific recipe, no OOM/NaN, full dataset permissions. Run short preflights **only on verified released GPUs** after dependency completion; CPU-only schedule-contract tests can be prepared now without touching live GPUs.

## Part 3 — Queue after the FIRST fully finished campaign

The user wants Q4 **behind whichever currently running experiment task finishes first**, not tied to a randomly selected host or interrupted queues.

1. Inspect present `QUEUE_STATUS.md`, controller.log, assignment matrices, tmux/processes and actual GPU occupancy on gpu01/gpu02. Recalculate completed/running/queued/failed for each current campaign (including the RSNA 4–5 extension). Estimate time-to-full-queue-completion from measured valid per-run durations, remaining steps and queue depth; do NOT guess from queue length alone. If ETAs unreliable, choose the first queue that *actually* becomes fully complete and releases its GPU pool.
2. Prepare Q4 run manifest NOW (only missing cells), dedicated new worktree and artifact root such as `/nobackup/enzez/q4_remaining_dense_20261119/artifacts/q4_remaining_dense_20261119/`. Use no GPU to develop CPU schedule checks, dataset/source audit or detached scheduler; never change working source/checkpoints from other campaigns.
3. Create a robust **single-instance detached dependency watcher / queue handoff** on a permitted existing login host. It should periodically check the four eligible pool statuses through SSH/read-only files, including truly finished key counts, old controller quiescence and independently verified empty GPU VRAM/process lists; use a file lock and idempotent checks to avoid double dispatch. Frequency low enough to avoid stressing the server (e.g. every 5 minutes). Record chosen pool and why (actual completed vs ETA) and leave other queues untouched. Do not treat a dead controller with pending jobs as 'completed'. If remote access fails or old campaign requires manual repair, stay safely queued and report.
4. **Only when the chosen pool's entire previous campaign has COMPLETED_VALID (or resolved all planned keys with no pending/active jobs in a separately authorized completion policy), its controller is disarmed and all its assigned GPUs are free**: preflight on those GPUs and launch Q4 missing-run queue detached with one trainer per free GPU, verified PID/UUID/advancing finite loss. Never steal gpu02 4–5 while RSNA controller is active. Never use gpu03/gpu04 for this specific request.
5. Do not let watcher run forever if zero eligible/missing runs: if `needs_new_run=0`, stop without launching and report Q4 gap coverage complete. If any blocking source/license/protocol discrepancy, mark that specific cell blocked; do not silently substitute methods or another dataset. The watchdog may keep waiting for a valid pool, but must not silently begin a changed scope after user request.
6. Store all metrics and detailed logs privately on cluster; do NOT send to public GitHub. Produce `Q4_QUEUE_HANDOFF_STATUS.md` with complete missing matrix, dependency target, actual chosen host/GPU IDs or estimated candidates, watcher controller/log PID and tmux, current queued vs running counts, instructions for manual monitoring. If new work has not yet started because existing queues are busy, explicitly say 'Q4 prepared/queued behind X; 0 new Q4 trainers started.'

**Important distinction:** scheduling a Q4-dependent watcher now is authorized, but it does not mean Q4 training has launched. State which phase really happened.

## Codex final response requirements

Return (A) **Q3's completed English rebuttal paragraph + verified table** with exact supported counts and official-source provenance, as a user-visible deliverable; (B) Q4 source audit summary (existing/missing/queued/running and total 144 examined), which concrete keys really need new runs, controller/watcher status, whose queue completion is the dependency, and monitoring commands; (C) explicit confirmation no live experiment or protected GPU/worktree interrupted. If unresolved, specify the exact absent evidence, not made-up counts. This is a current action, not a promise to perform later by ChatGPT.
