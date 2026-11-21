# Codex: HPL supervision-frequency x uncertainty/bilevel-mechanism factorial (RUN EXPERIMENTS, guarded)

## Language and authorization

**All Codex CLI communication, progress updates, markdown/CSV report content, and final results MUST be in ENGLISH.** The user's ChatGPT conversation is Chinese; do not switch the CLI to Chinese.

The user **explicitly authorizes a new HPL-focused mechanistic experiment**, and has substantial GPU capacity. This task is authorized to implement and run *only the scoped, prevalidated new experiments described here*. It is NOT permission to change, rerun, interrupt or relabel existing campaigns.

**Research question:** How much of the behavior of the official HPL regression method is attributable to its **already-dense labeled predictor supervision** versus its **learned heteroscedastic uncertainty and bilevel outer mechanism**? Do not presuppose that the supervision schedule dominates. We need a clean empirical decomposition; publication/rebuttal claims must follow the results.

## Context you MUST read before implementation

1. Prior private audits on gpu04:
   - `artifacts/hpl_historical_source_audit_20261120_130602/HPL_OFFICIAL_VS_ADAPTED_SOURCE_DIFF.md`
   - `artifacts/hpl_historical_source_audit_20261120_130602/HPL_EXISTING_RUNS_INVENTORY.csv`
   - `artifacts/hpl_historical_source_audit_20261120_130602/HPL_W1_RESPONSE_AND_GAP.md`
   - `artifacts/hpl_official_baseline_fairness_audit_20261120/HPL_FAIRNESS_DECISION.md`
   - `artifacts/hpl_official_baseline_fairness_audit_20261120/HPL_OFFICIAL_BASELINE_CANDIDATES.csv`
   - `artifacts/hpl_official_baseline_fairness_audit_20261120/HPL_MINIMAL_NEXT_ACTION.md`
2. Official HPL `sxq11/Heteroscedastic-Pseudo-Labels` at **`89f9f8bd467a0d3f81a8ada8708c3fe4fe31ca20`** and its NeurIPS 2025 methodology / ablation definitions. Official UTKFace: epoch follows unlabeled loader and cycles labeled batches, with `ssl_mult=2` dataset duplication; official IMDB: labeled-controlled loop. These differ by dataset. Both official methods also have their own hyperparameters; do NOT label our adapted results as unchanged official reproductions.
3. Existing paper regression adapter source/protocols, including HPL's `higher` inner step (head only), uncertainty learner and optimizer, meta-update (every five main predictor steps), weak/strong pseudo-labels, validation checkpoint selection, normalization, RNG, and dataset manifests.
4. Historical "Original" UTK benchmark used only ~59 supervised predictor steps and ~12 meta updates vs 563/113 in adapted dense. This is NOT a valid equal-budget comparison. Existing old UTK HPL k-sweep has *effective labeled repeat=1*, not 2, and incomplete executed-code provenance; do NOT silently substitute it for our factorial.

Never use evaluation/test labels in an outer optimization batch. Any meta labeled batch MUST originate from the training labeled subset, not the validation or held-out test set. Use validation strictly for checkpoint selection.

## The pre-registered main design

**Two datasets:** UTKFace and IMDB-WIKI, both 5% labeled (first full pass; deliberately includes the historical positive and negative HPL response).
**Two backbones:** ResNet-50 and DINOv2-S/14.
**Three matched seeds:** 0,1,2. Do not choose seeds after looking at outcomes. A six-seed expansion is only a later option, not the initial commitment.
**30 epochs**, same canonical split/augmentation/normalization/model initialization per paired seed, with saved best validation checkpoint and one held-out test evaluation per completed run.

**Factor S: predictor supervised-anchor schedule**, holding the FULL unlabeled traversal, main predictor update count, and 30-epoch horizon fixed:
- `sparse_even_native_mass`: EXACTLY M supervised predictor updates among U total predictor steps, evenly distributed over U positions. Compute `M=len(native labeled loader)` after native `ssl_mult=2` duplication, `U=len(unlabeled loader)`, and define zero-based positions `p_j=floor((j+0.5)*U/M)` for j=0,...,M-1 (check uniqueness, bounds, spacing, exact count; if U<M, stop and revise). Use each native labeled batch exactly once in this sparse regime unless official loader semantics require otherwise. The previous 5% as-run benchmark had approx UTK M=59, U=563; IMDB M=598, U=5685. Verify per actual data/loader, do not hardcode.
- `dense`: supervised predictor MSE in **every** one of the U steps, recycling labeled batches as needed, while otherwise following the exact same unlabeled batches and optimizer updates.

The frequency effect (dense vs sparse) changes the **number of labeled predictor exposures and supervised-loss mass** by design. Do NOT claim this isolates temporal placement with fixed anchor mass; do NOT call it a same-labeled-exposure comparison. A further mass- or exposure-matched extension is OPTIONAL and must be separately labeled/proposed, not silently claimed.

**Factor P: pseudo-label/uncertainty mechanism**, all sharing EXACTLY the same weak-to-strong consistency target, predictor architecture/data, U main optimizer steps, Ls definition, and shared optimizer/hyperparams as much as possible:
- `P0_uniform`: uniformly weighted unlabeled consistency, no learned uncertainty net and no bilevel meta. Preserve existing Uniform regression branch semantics (including chosen unlabeled coefficient) and document any weight-scale differences relative to HPL. This is not intended as "HPL without only the outer loop"; it removes both uncertainty weighting and bilevel learning.
- `P2_full_HPL`: learned heteroscedastic uncertainty weighting with the **authentic HPL differentiable inner adaptation and bilevel outer learner**, followed by native predictor gradient step. Keep HPL per-5-predictor-step meta cadence and **the same number of meta updates in sparse and dense conditions** (approx UTK 113 and IMDB 1137), independent of S. Keep learner loss/auxiliary optimizers and gradients as authentic as possible.

This 2x2 core is **2 datasets x 2 backbones x 3 seeds x 2 anchor schedules x 2 mechanisms = 48 runs**.

**Strongly preferred explicit outer-loop ablation, P1 (conditional):**
- `P1_uncertainty_no_BLO`: an HPL **uncertainty-learner without bilevel optimization** condition, using the **published HPL no-BLO ablation's training rule** or the exact supported official-code implementation if available. Preserve the learned uncertainty mechanism and match P2's loss weights/head/predictor update structure where scientifically definable.
- Before running P1, inspect the actual published no-BLO ablation and code, write the exact first-order learner objective, its gradients, optimizer cadence and labeled-example usage into the experiment manifest, and run targeted unit tests showing the learner receives nonzero intended gradients while no differentiable bilevel `higher` path occurs. **Never use a randomly initialized frozen uncertainty net and label it "no outer loop".**
- If a defensible published P1 implementation can be established, run **24 additional P1 runs** (2 datasets x 2 backbones x 3 seeds x 2 schedules): then the full main study is a **2x3 factorial, 72 runs**.
- If published no-BLO training semantics cannot be faithfully implemented and tested quickly, **still launch the independently valid 48-run P0/P2 core**, mark P1 `BLOCKED_METHOD_FIDELITY`, and explain what would be needed. Do not invent a P1 rule or overclaim isolation of "outer loop".

## Crucial HPL meta/supervision confound to prevent

A preserved benchmark code variant selects `x_meta=x_lb` on supervised predictor anchor steps and samples `x_meta` from a separate loader on no-anchor steps. This changes meta-example routing when S changes, despite equal per-five-step meta count. **Do not permit this to undermine schedule isolation.** Choose a scientifically explicit **independent, fixed-seed training-labeled meta stream** on *all* meta-update steps for both sparse and dense HPL schedules (consistent with the HPL outer evaluation concept), and keep the auxiliary `unc` labeled stream independent and identical across S. Record and expose the difference between this controlled HPL regression adaptation and old paper runs; the new factorial is internally matched, not directly identical to published Table 4 results.

**Further:** A "full-HPL vs Uniform" comparison necessarily differs in auxiliary/meta exposure. It measures the combined gain of uncertainty weighting + bilevel learned feedback, not an outer-only effect. Only a valid P1 comparison can address the added value of BLO beyond a learned uncertainty learner. Explicitly count and report main predictor labeled samples AND auxiliary labeled samples (meta/unc) separately; never call all cells labeled-exposure matched.

## Phase 0: preflight and falsification tests (required BEFORE full queue)

Implement in a NEW isolated local worktree / new output namespace, leaving the public release and every existing campaign unchanged. Version-control or checksum exact executable trainer source and config.

Produce a preflight matrix with actual (U,M), main steps=U, sparse anchors=M, dense anchors=U, HPL meta steps=ceil(U/5), native `ssl_mult`=2 and effective labeled loader lengths for BOTH datasets/backbones; trace **first epoch** schedule positions and selected sample IDs.

Unit/smoke tests (synthetic + minimal real batches; not formal runs):
1. Sparse and dense have identical U optimizer updates, unlabeled batch ordering, learning-rate schedule, number/cadence of HPL meta/unc optimizations, and `w_ulb` across S for a given P.
2. Label split/manifest, initialization and core model weights match across cells within the same dataset/backbone/seed; separate RNG streams avoid data-order drift when branches use different numbers of forward passes.
3. `P2` actually gives nonzero gradient from labeled meta signal through differentiable inner adaptation to `unc_learner`; `P0` has no such learner and uses the intended constant uniform weights. `P1`, if implemented, uses valid learned uncertainty with no outer higher gradient.
4. Auxiliary HPL meta/unc training-labeled streams do not depend on whether the current main step is sparse or dense supervised.
5. Only train subset labels used for supervised predictor and HPL meta/unc optimization; validation for checkpoint selection, test evaluated ONCE after frozen val-best selection.
6. Verify `ssl_mult` is truly effective repeat=2 for all cells and no old repeat=1 k-sweep contamination.
7. Check loss scale (constant-weight Uniform vs HPL exp(-u)/2), logs for possible near-zero consistency or explosion; avoid post-hoc tuning based on test.
8. CPU static/run validation before GPU allocation; use small GPU smoke only on idle, approved GPUs. If severe method-fidelity/protocol issue exists, stop BEFORE submitting full queue and explain; do not quietly change factors.

**Only when Phase 0 passes:** launch new formal 48-run P0/P2 core; launch P1 24-run extension **only if** its fidelity preflight passes. The user has authorized these experiments. Do not require another user message simply to launch validated core. Do not add additional datasets/label ratios/6-seed expansions without explicit approval.

## Compute resources / non-interference

The user has a large GPU fleet. **Dynamically inspect current GPU free memory, utilization, host allocations and active process ownership**; never rely on stale free-card reports. Prefer idle A40 hosts/gpu04 and gpu05 as appropriate, else free 2080 Ti; pick safe per-job GPU memory fit (especially DINO on 2080 Ti). Leave gpu03 service GPUs alone. DO NOT kill, reschedule or preempt existing workloads, especially Q4 TE, MT/TE ablation or other users. Start only when capacity is verifiably free; if not, safely queue without preempting. Respect disk quota, output paths, timeouts; checkpoint and resume only from your own isolated new runs. Validate all run IDs after launching and use existing campaign monitor patterns without interfering.

## Analysis (pre-register contrasts, report full outcomes)

Report test MAE (↓) mean±population SD (ddof=0) for each setting; paired per-seed MAE differences and exploratory 95% paired t CIs for n=3 (explicitly wide and uncertain). Validate all planned seeds, errors, configs, source hashes, final-test one-time guard.

For each dataset/backbone, define:
- `S_effect(P) = mean_seed[ MAE(P, sparse) - MAE(P, dense) ]`. Positive = dense predictor supervision helps under mechanism P.
- `HPL_effect(S) = mean_seed[ MAE(Uniform, S) - MAE(Full_HPL, S) ]`. Positive = HPL mechanism bundle helps at schedule S; this bundles uncertainty weighting and bilevel plus auxiliary labeled data.
- `interaction = S_effect(Full_HPL) - S_effect(Uniform)`. Difference in differences: whether learned uncertainty/BLO changes dependence on supervision density.
- **If P1 valid**: `BLO_increment(S) = mean_seed[ MAE(UL_no_BLO,S) - MAE(Full_HPL,S) ]`. This isolates BLO more closely than P0 vs P2, subject to documented differences in auxiliary learner loss and exposure. Also compare UL_no_BLO vs Uniform at fixed S.

Do NOT use significance to overclaim with only 3 seeds. Say whether (a) dense supervision helps under both methods, (b) relative mechanism gains shrink/expand with density, (c) dataset/backbone interactions reverse effect. Do not turn correlation into an unsupported claim that HPL paper's gain "really" came mostly from supervised density; original HPL ablations may already support bilevel. Provide a truthful, short English paper-ready interpretation and clear limitations.

Optional supporting diagnostics on EXISTING training data and checkpoints (avoid extra test evaluations): label supervised / unlabeled loss trajectory, meta gradient magnitude, weight mean/variance and effective lambda, train-only pseudo-label error proxy if ground-truth labels are actually available in the **train-labeled** subset. Do not compute hidden unlabeled ground-truth labels as a training or selection signal. Diagnostics must not affect model selection or run definitions.

## Deliverables (private local only)

Create distinct new private folder `artifacts/hpl_factorial_schedule_vs_bilevel_20261120/` (or unique timestamped child), with:
1. `PRECHECK_AND_PROTOCOL.md`: exact implemented P1 (if any), pinned source/trainer SHA, budget/method controls, RNG, validation, accepted deviations from upstream, GPU assignments.
2. `RUN_MANIFEST.csv` and `STATUS.md`: row per factorial cell, seed, host/GPU/job, log/artifact paths, method P, schedule S, M/U, epochs, status/error; timestamp.
3. `FACTORIAL_RESULTS.csv`: per-run and aggregate metrics, paired contrasts/CI, units, effect signs; never fabricate uncompleted cells.
4. `HPL_MECHANISM_SUMMARY.md`: mechanism attribution, faithful original published no-BLO semantics, what the experiment does and does not identify, short English figure/table-ready response.

Final CLI response **IN ENGLISH ONLY** should give: preflight PASS/FAIL, conditions launched/blocked, exact job counts, hosts/GPUs, live vs complete vs failed, artifact paths, no-interference confirmation and concise scientific interpretation of available results. Do NOT claim jobs complete before verifying them.

## Public GitHub confidentiality

This GitHub branch contains **instructions only**. Do not commit/push any unpublished numerical results, model checkpoints, run paths containing protected patient identifiers, raw data, reviewer content, draft rebuttal, private logs or credentials here. Local private run outputs remain on the authorized lab host.
