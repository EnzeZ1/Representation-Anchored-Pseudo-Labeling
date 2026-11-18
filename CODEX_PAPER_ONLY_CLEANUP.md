# Codex Task — Make a clean *paper-only* source release

## Goal
Create a **new, minimal, scientifically faithful paper-only code release** for the user-provided under-review manuscript (22 pages). A private PDF is NOT included in this public repository. The figure/table and method inventory below was extracted from that manuscript and defines the public task scope. The agent must ask for private access to the PDF before asserting it verified exact paper-text coverage. Do not assume that unrelated code in the original repository belongs in this paper. Work on the code, not the manuscript or experimental claims.

**Repository:** `EnzeZ1/Representation-Anchored-Pseudo-Labeling`

**Source snapshot / immutable reference:** branch `code/pseudo-labeling-essential-20261117`, directory `essential_code/`.

**NON-NEGOTIABLE:** You are reading this task on a staging/task branch. Create a **separate new working branch** for the paper-only edition, e.g. `code/pseudo-labeling-paper-only-release-20261117`, preserving the task branch, reference branch, `main`, all original server files, and the local source archive **unchanged**. No force-push, destructive overwrite, loss of provenance, training rerun, or silent behavior changes. The `shared/` directory is deduplicated storage across source profiles—not one executable combined recipe. Reconstruct each original run using `essential_code/profiles.json` and `essential_code/SOURCE_INDEX.csv`, never by arbitrarily mixing `shared/` and `variants/`.

## Exact scope: include ONLY what is in this manuscript (main body + appendices), plus its *necessary* dependencies

### Methods and experimental setup
- Six SSR objectives **each with Original and +Interaction / +Reorient exactly as actually run**: (1) Plain (uniform weak-to-strong consistency), (2) Temporal Ensembling (TE), (3) Mean Teacher (MT), (4) Heteroscedastic Pseudo-Labels (HPL), (5) UCVME, (6) SimRegMatch.
- Four datasets and their **protocol/split-generation code (not private data)**: UTKFace, IMDB-WIKI-DIR, RSNA Bone Age, and CheXchoNet LVIDd.
- Only the two paper backbones: pretrained ResNet-50 and DINOv2 ViT-S/14. Keep the exact optimizer, transform, label-normalization, seed, checkpoint-selection, and other method-specific code needed to match the paper.
- The paper's main benchmark is **5% labels, 30 epochs, six seeds (0–5)**; analyses typically use **seeds 0–2**. Do not include unrelated 10%/20% sweep campaigns merely because they exist on the servers (unless a specific paper result actually depends on them). Rebuttal-only experiments belong outside this *paper-only* package.

### All paper figures and tables must be traceable
Include training/diagnostic/aggregation/plotting code, where it exists, for:
- **Figure 1 / Table 8:** HPL, UCVME, SimRegMatch uncertainty-weight vs. pseudo-label-error Spearman correlations.
- **Figure 2 / Table 9:** TE and MT pseudo-target error and temporal drift, including correct seen-example handling.
- **Figure 3:** anchor-schedule illustration source if code-generated; otherwise document that it is schematic.
- **Figure 4:** augmentation-view choices (same-view, weak–weak, weak–strong) crossed with supervised anchor interval.
- **Figures 5 and 8:** local gradient geometry, reference loss, vectors and interaction gain.
- **Figure 6:** supervised-frequency sweep `k = {1,2,4,8,20}`; reference-task diagnostic; fixed-budget front/even/back placement (29 supervised positions within 563 unlabeled steps), and its exact schedule definitions.
- **Figure 7:** supervised-only, supervised-only recycled, Plain/SSR, and dense-interaction labeled-exposure/update controls.
- **Table 7:** EVEN-29 versus dense coefficient-mass-matched (coefficient `29/563` each of 563 steps), preserving the distinction between *nominal coefficient mass* and actual sample exposure.
- **Tables 1–6 / 12–14:** all six methods × four datasets × two backbones; test MAE/R² and paired seed-level comparisons. Include the aggregation scripts when known, not all raw logs/checkpoints.
- **Tables 10–11 and Appendix C:** dataset protocols, hyperparameters, model/dataloader definitions and reproducibility instructions.
- **Appendices B–E:** all relevant diagnostics, settings, comparisons and derived plots/tables. The theory in Appendix A does not require executable code unless a figure or numeric example depends on it.

### Explicitly EXCLUDE from the new paper-only release (after verifying they are not imported by retained code)
- STSB / text-regression experiments, RoBERTa and BiLSTM, text augmentation.
- KonIQ10K or unrelated image-quality experiments.
- DINOv3 model/experiments, RankUp experiments/patches (the paper cites some methods but does not experimentally run them). If an existing paper trainer imports such modules unconditionally, clean the import path **without changing numerical behavior**, or keep a minimal dependency temporarily and clearly document why.
- Unrelated label-ratio sweeps, exploratory prototypes, dead launchers, administrative process monitors, operational repair/history scripts and duplicate snapshots not needed for any paper result.
- Datasets/images, CheXchoNet patient files, restricted manifests, checkpoints, predictions containing protected information, full logs, queue state, results archives and virtual environments.
- Historical versions not used in the paper. **Keep exactly as-run historical versions that ARE used in the paper**, including Original versus +Interaction; when versions differ, label their purpose clearly rather than merging them.

## Specific issues found in the archival audit: verify and expose, not silently rewrite
1. **Plain:** the Original training loop consumes a labeled iterator without cycling after exhaustion; the +Interaction loop recycles it. Source: `variants/plain_original_formal/training/train.py` versus `variants/plain_reorient_reference/training/train.py`. The former trainer's CLI default for `weighting_intervention` is `rapl`, while the paper's Plain is **uniform**. Track the **actual historical command/config** (e.g. any explicit `rapl_uniform_weight`) before labeling a run as Plain; do not assume defaults.
2. **MT and TE:** their native loops already contain supervised updates; the +Interaction versions add an **extra supervised MSE term**, rather than merely changing the frequency of a previously absent loss. Preserve that fact in README and code mapping. Do not claim the objective is identical across versions.
3. **HPL:** trace the *actual* Original and +Interaction versions for **all datasets**. The existing `hpl_reorient_medical_formal10` profile currently does not clearly identify a +Reorient medical trainer entrypoint. Find it by actual provenance, or mark it missing; never substitute a different trainer. Check original labeled sampling/`ssl_mult` and native HPL recycling details.
4. **SimRegMatch:** the archived as-run implementation has `np.percentile(..., q=0.95)`, which means the 0.95th (not 95th) percentile. The historical launcher validates `0.95`. **Do not silently change to 95 in the as-run source**; document the exact behavior and list a corrected variant only if explicitly authorized as a new experiment.
5. **Figure 6 vs Table 7:** distinguish the front/even/back placement definition and the separate EVEN-29 mass-matching run, with their respective seeds, checkpoint/aggregation settings; never equate results from different protocols.
6. **Deduplication:** the current 152 stored paths have 143 distinct SHA256 values. At least nine extra byte-identical copies remain across five SHA groups (notably RSNA/CheXchoNet HPL snapshot paths). Remove duplicate payloads in the **new** branch, but preserve alias-to-SHA mappings. Never remove an actually distinct historical version just because filenames are similar.
7. Missing or ambiguous source for Figures 1, 2, 4, 5, 8 or any table should be reported honestly as **NOT VERIFIED / NOT FOUND**, not reconstructed based on expected results. Search existing provenance/original archives when available; do not fabricate a pipeline or infer which configuration produced a published number.

## Implementation approach and safeguards
1. Use the inventory below and, if privately supplied, parse the manuscript to construct a **paper-to-code coverage matrix** (Figure/Table → original method and condition → exact source profile, SHA256 and script → command/flags → dataset manifest protocol → aggregation/plot script → status).
2. Build a new `paper_release/` in a **new branch** with clean directories (e.g. `methods/`, `data/`, `models/`, `experiments/`, `analysis/`, `configs/`, `provenance/`), choosing the smallest executable dependency closure. Avoid creating an artificial merged version that did not produce the results; prefer per-method, per-condition exact sources when necessary.
3. Keep third-party notices, license conditions, upstream repository and commit references. In particular, keep the Mean Teacher `TwoStreamBatchSampler` dependency or an exact documented equivalent; don't remove licenses while pruning.
4. **No silent scientific changes**: no changes to losses, scheduling, sample-selection policy, training budget, seeds, model initialization, optimizer, benchmark parameters, numerical precision, or data protocol. If technical changes are necessary to make relocated files importable, clearly separate such portability-only modifications from preserved source and log their diffs/checksums.
5. Run static checks (`compileall` / AST parsing, import/CLI smoke tests, import closure, manifest/hash checks). Run synthetic-data smoke tests only if possible without changing code semantics. State explicitly when missing datasets/checkpoints prevent reproducing MAEs. Do **not** retrain the paper while doing a code cleanup.
6. Do not publish the manuscript, personal identities, private data or an issue tying an anonymous under-review submission to a personal public GitHub account. Work on a non-public workspace or private branch/repo if available; if the current repository is public, avoid publicly mentioning Submission 99 or author identities in commit messages, public Issues and PR text. Ask the user before an external/public paper release.

## Deliverables / acceptance criteria
- **New clean paper-only branch**; archive/source reference unchanged and retrievable.
- `README.md`: scope, layout, dependencies, intended invocation, datasets/pretrained weights (external), limitations and environment setup.
- `PAPER_TO_CODE_MAP.md` or `.csv`: **every** paper Figure 1–8 and Table 1–14 accounted for as reproduced, script located, or missing—not simply claimed.
- `AS_RUN_METHOD_MATRIX.md`: for six Original and +Interaction implementations, identify per-step `L_s`, `L_u`, extra supervised terms, optimizer steps/epoch, label recycling, method-specific differences.
- `SOURCE_MANIFEST.csv`: path, purpose, originating profile, exact SHA256 and upstream/provenance, with duplicate aliases documented.
- `MISSING_AND_UNVERIFIED.md`: any missing scripts, hardcoded external dependencies, unclear commands, license limitations, reproducibility blockers and science-claim mismatches.
- `CLEANUP_SUMMARY.md`: before/after file counts, size, what was excluded and why, retained technical differences, hash/static checks passed/failed, exact commands tested.
- No unneeded source, snapshots, private assets, data, logs or checkpoints in the paper-only directory; no scientific behavior changes; no claims of validated MAEs without actual controlled reproduction.

**Priorities:** correctness/provenance and coverage of published work > minimal file count > cosmetics. Please finish with a clear audit report and only then propose (do not perform) any scientifically substantive fixes or reruns.