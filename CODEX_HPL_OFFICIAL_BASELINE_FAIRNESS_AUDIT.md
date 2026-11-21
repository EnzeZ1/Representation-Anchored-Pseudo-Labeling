# Codex task: HPL OFFICIAL-BASELINE FAIRNESS AUDIT (READ-ONLY)

**All CLI replies, status updates, reports, and final summaries MUST be in ENGLISH.** This rule also applies if the initiating ChatGPT conversation is in Chinese.

## Goal

Before deciding whether to run any more HPL experiments, establish whether the paper's "Original HPL" baseline and the existing +Interaction results fairly represent the **actual official HPL training protocol**, and whether any **already completed runs** can serve as a faithful, matched official-baseline comparison. We must not needlessly repeat previous experiments or misrepresent code provenance.

**READ-ONLY AUDIT ONLY. NO TRAINING AUTHORIZED.** No code edits, no new queues, no inference, no GPU preflight, no reruns, no stopping/restarting other jobs. Inspect source, history, configs, checkpoints metadata, and prior logs/artifacts only. Output private local audit reports; do not put unpublished numerical findings or manuscript/reviewer material onto public GitHub.

## Starting point: prior completed audit

Read the previous private report on gpu04 if available:
- `artifacts/hpl_historical_source_audit_20261120_130602/HPL_OFFICIAL_VS_ADAPTED_SOURCE_DIFF.md`
- `artifacts/hpl_historical_source_audit_20261120_130602/HPL_EXISTING_RUNS_INVENTORY.csv`
- `artifacts/hpl_historical_source_audit_20261120_130602/HPL_W1_RESPONSE_AND_GAP.md`

Previous public task instructions are on `tasks/hpl-historical-source-audit-20261120/CODEX_HPL_HISTORICAL_SOURCE_AUDIT.md`.

Previous audit findings to **verify from actual sources before asserting**:
- Pinned upstream repository: `sxq11/Heteroscedastic-Pseudo-Labels`, verified upstream commit `89f9f8bd467a0d3f81a8ada8708c3fe4fe31ca20`.
- Official **UTKFace** HPL uses `len(unlabeled_loader)` and cycles the labeled iterator; official **IMDB-WIKI** HPL uses `len(labeled_loader)`; both duplicate labeled data with `ssl_mult=2` by default. These official dataset-specific implementations DIFFER.
- Historical Codex record indicates a September 7 change to a *local* short-loop UTK "Original" trainer, including modifications inside a local directory called `-upstream`. That locally named directory must NOT be conflated with untouched upstream source.
- In our as-run UTK 5% regression benchmark: "Original HPL" was **59 predictor updates / 59 supervised-containing updates / 12 meta updates per epoch**, whereas +Interaction was **563 / 563 / 113**. IMDB 5% was **598 / 598 / 120** versus **5685 / 5685 / 1137**. Thus these A/B contrasts ALSO change total predictor updates, unlabeled traversal, and meta optimizer budget.
- Existing valid A/B benchmark runs cover UTKFace and IMDB-WIKI at 5% on R50 and DINOv2-S for seeds 0–5. Earlier UTK HPL 563-step frequency sweep `k=1,2,4,8,20` (seeds 0–2, two backbones) has a different effective labeled repeat (`1`) and does not automatically recreate the original native `ssl_mult=2` baseline. Archived source hashes may be missing.
- Benchmark IMDB HPL is a UTK-derived regression adaptation rather than a literal port of official IMDB HPL; settings such as `lambda2=0.1`, batch size and weight decay differ from official IMDB.

## Audit questions (answer each definitively or mark UNVERIFIED)

### A. Exactly what does the official baseline do?

For each **official UTKFace and official IMDB-WIKI** trainer at the pinned upstream commit, establish the actual epoch loop budget, labeled iterator behavior, `ssl_mult` semantics, total supervised predictor update count, meta/uncertainty learner frequency and update count, combined predictor loss and update cadence, learning rates, optimizer(s), weight decay, batch size, weak/strong augmentations, dropout, scheduling, and checkpoint selection. Use concrete paths, line ranges, and commit IDs.

Note crucial subtleties: a labeled dataset repeated twice is not the same as cycling a labeled iterator over the entire unlabeled loader; a per-5-step meta update may use labeled data even when a main supervised anchor is absent; original HPL algorithm code is not automatically equivalent to our scalar-regression benchmark port with different models and data.

### B. Are existing published-benchmark results faithful to official code?

Compare **exact as-run** trainer revision and config for:
1. Official upstream UTK HPL and its closest **benchmark-compatible faithful regression port**.
2. Original short-loop HPL in the paper (59 UTK / 598 IMDB).
3. HPL +Interaction (563 UTK / 5685 IMDB).
4. Old UTK 563-step HPL frequency `k=1,2,4,8,20` (particularly k=1), plus any archived faithful official-loop controls.
5. Original official IMDB-WIKI HPL and our UTK-derived IMDB implementation.

Report exact differences in **U, L, optimizer steps, supervised predictor steps, labeled exposures, meta updates, `ssl_mult`, loss weights, optimizer/augmentation protocols, PyTorch versions, seed/manifest identity**, and changes to HPL objective itself, if any.

Compare only like with like. No claim of a fair official-baseline victory if as-run configs differ in material ways. Explain whether maintaining official full-U loop and `ssl_mult=2` would already reproduce the **conceptual k=1 intervention** on UTK, and precisely what would still differ in our current benchmark.

### C. Inventory existing faithful-baseline candidate results

Search authorized active and historical results/archive paths on accessible hosts READ ONLY, including:
- Official/unmodified upstream runs; local `-upstream` variants (check hashes rather than names).
- Existing HPL Original / +Interaction, frequency sweep, dense/clean-dense, native loop variants.
- Benchmarks UTKFace and IMDB-WIKI, 5% labels, R50 and DINOv2-S, seeds 0–5 (report at least 0–2 distinctly).
- Checkpoint/config/metadata provenance, whether test was computed once, validation selection, historical framework/runtime differences.
- Particularly look for *official UTK full-U + labeled recycling* results that would remove the questionable 59-step comparator **without running anything new**.

Classify each candidate as:
- `FAITHFUL_UPSTREAM` (literal original training code, only if actually supported);
- `FAITHFUL_REGRESSION_PORT` (official HPL schedule/objective faithfully retained, dataset/backbone changed transparently);
- `BENCHMARK_SHORT_LOOP`;
- `BENCHMARK_DENSE`;
- `FREQUENCY_OR_OTHER_CONTROL`;
- `INCOMPATIBLE_OR_UNVERIFIED`.
For inaccessible archives, say `UNVERIFIED_ARCHIVE`; do NOT conclude absence merely from failed SSH. Do not cherry-pick seeds.

### D. Does the paper need any NEW HPL run?

Give a decision table:
1. Can we use existing results for a **faithful official UTK HPL** comparison, yes/no/uncertain? If yes provide exact files and matching limitations.
2. Can existing HPL k=1 frequency results serve as that baseline? Why/why not, with numeric update/exposure differences.
3. Does IMDB-WIKI need a comparable official-implementation control, or would a clearly labeled regression adaptation be the scientifically appropriate baseline?
4. What is the **minimum** additional experiment set needed to address HPL fairness if no suitable valid runs exist? Give precise dataset/backbone/seeds, source changes required to make a faithful *regression-port* original, training budgets, and estimated run count — a proposal only.
5. Is the prior proposed C59/C598 sparse-extended 12-run ablation useful or lower priority than a genuinely faithful upstream-loop comparator?

Do not treat a revised source as an already-executed result. Separate *evidence-based answers*, *inferences*, and *open uncertainties*. If a new baseline would show zero change from +Interaction because the official UTK baseline already has a dense loop, state this explicitly and interpret the paper's originality claims accordingly.

## Required private outputs

Write into a new private audit-only folder (do not overwrite prior reports):
- `HPL_FAIRNESS_DECISION.md` (executive yes/no/uncertain assessment; source and result provenance; exactly what needs correcting in Appendix C.6 and Table 4 narrative)
- `HPL_OFFICIAL_BASELINE_CANDIDATES.csv` (each existing run: host/path, dataset, backbone, seed, verified trainer SHA, code category, U/L, supervised/total predictor and meta updates, `ssl_mult`, MAE, comparability flags)
- `HPL_MINIMAL_NEXT_ACTION.md` (short, prioritized *no-run* proposal, if anything is missing)
- Optionally a **defensible short English rebuttal paragraph** for reviewer eQjg W1, acknowledging official UTK recycling and UTK short-loop benchmark mismatch.

Final CLI response **IN ENGLISH ONLY**:
- `Official UTK HPL training loop:` ...
- `Our UTK paper Original loop:` ...
- `Existing faithful official-loop results:` ...
- `Can k=1 archived runs substitute?` ...
- `Does a new experiment actually need to be run?` ...
- `Files:` ...

## Strict operating rules

- Absolutely **NO** new training, queued/requeued jobs, GPU inference, test reruns, code or configs modifications, experiment restarts/stops, checkpoint mutation, or overwrites to results.
- CPU-only file metadata, checksum, static source inspection, git diff/log, CSV/log parsing are allowed.
- Do not disrupt the existing TE Q4, MT/TE supervision ablation, other SSR jobs or shared GPU infrastructure.
- Public GitHub files must contain ONLY generic instructions; no unpublished results, raw patient data, manuscript, reviewer identities or private logs.
- All communication to the user **in the CLI must be English**, even if the user talks to ChatGPT in Chinese.
