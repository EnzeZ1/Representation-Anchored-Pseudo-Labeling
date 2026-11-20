# Codex — Execute 24 RSNA 5% UCVME fixed-budget scheduling experiments on gpu05 GPUs 4–7

## User authorization, objective and limits

The user explicitly approved **launching** this exact 24-run experiment using **only physical GPU IDs 4,5,6,7** on `instgpu-05.cs.wisc.edu` (reported to have four usable A40 GPUs). This document is a handoff for the user's live Codex session. **Writing it to GitHub has not invoked Codex or launched anything**. When invoked, connect and check actual availability, source/dataset/protocol coherence, run CPU/small GPU preflights, then start a detached restartable queue and verify live training PIDs. Do not request another confirmation for this exact matrix if preflight passes.

**Scientific question:** does the *temporal placement* of a fixed number of supervised updates matter in a **published existing SSR mechanism** (UCVME's heteroscedastic two-model consistency), not just Plain/Uniform, on RSNA Bone Age at 5% labels? This is a complementary extension, not a replacement for the already completed RSNA 5% Plain 48-run campaign on gpu04. Do not claim that UCVME uses Plain's pseudo-label loss.

## Remote GPU and safety requirements

```bash
ssh enzez@instgpu-05.cs.wisc.edu
```

- **Exclusively GPU 4,5,6,7 on gpu05**; don't use other gpu05 GPU IDs and do not use gpu01–04. The user's reported hardware is **A40 (48 GiB)**; confirm by `nvidia-smi -L`, actual UUID, VRAM, process list, two checks of GPU compute/memory and current owners/queues. If a requested GPU is occupied or lacks VRAM, use only genuinely idle GPUs from {4,5,6,7}; if none usable, do not launch.
- One training process per physical GPU; `CUDA_VISIBLE_DEVICES=<physical_id>` with local `cuda:0`. Treat protected medical image access and quotas appropriately. No stopping, restarting, reassigning or modifying current user/other researchers' jobs, controllers, checkpoints or source trees. The previously finished RSNA Plain 48-run results on gpu04 and RSNA ratio extension on other hosts are **read-only**.
- Use a **new isolated local git worktree**, no destructive cleanup on existing checkout. Avoid privileged operations, downloads of protected datasets, or public publication of source-derived private records.
- Public GitHub task branch must contain instructions only; keep all unpublished numeric results, logs, manifests with sensitive image metadata, raw medical data, trained model weights and reviewer/manuscript details server-local.

## Exact approved run matrix: 24 jobs, no automatic expansion

- **Dataset**: original canonical **RSNA Pediatric Bone Age**, **5% labeled**. Use existing RSNA cohort and official seed-specific 0.05 manifests.
- **Method**: **UCVME**, **two separately trainable predictor networks** with native uncertainty/heteroscedastic branches and unlabeled consistency/variance losses. Not Plain; not TE or MT.
- **Backbones**: ResNet-50 and DINOv2 ViT-S/14.
- **Conditions**: `front`, `even`, `back`, `dense_mass_matched` (4).
- **Seeds**: 0,1,2 (3 paired seeds, no cherry-picking or extension to 3–5).
- **Epochs**: 30; full training from a fresh identical initialization across conditions for each backbone/seed.
- **Total**: 2 backbones × 4 conditions × 3 seeds = **24 distinct run keys**.

**Historical source**:
- `essential_code/shared/third_party/UCVME/ucvme_age.py` (RSNA-adapted UCVME source with two models, stochastic ensemble calculation, `run_epoch`, `reorient` and uncertainty losses); inspect the **actual archived as-run UCVME/RSNA 5% profile and its provenance/config** on authorized local disk, using `essential_code/profiles.json` and `essential_code/SOURCE_INDEX.csv`. Do not assume mixing all `essential_code/shared` files creates a coherent runnable version.
- Official upstream lineage: `xmed-lab/UCVME` at recorded `9f27a579` in the historical benchmark's provenance. This is an adapted regression benchmark, not an unmodified upstream executable.
- `essential_code/shared/data_processing/rsna_bone_age.py` and actual UCVME data loader path for RSNA manifest and label target scaler. The **RSNA Plain** schedule result/source may be used solely for anchor indexing, data cohort cross-check, and analysis protocol; **DO NOT import Plain's labeled MSE, batch size 8, optimizer, clipping policy, image transforms or unlabeled loss into UCVME without validating the UCVME as-run configuration**. Historical UCVME code generally has two models and `drop_last=True` for train; verify the precise RSNA UCVME recipe (batch size, num_workers, optimizer groups for each backbone, dropout, schedulers, w_ulb, samp_ssl, samp_fq, normalization, augmentation, reduction, output head and seed behavior) directly.

## Scientifically precise supervision gating

Let `U=len(actual_ucvme_unlabeled_train_loader)`, `L=len(actual_ucvme_labeled_train_loader)` at each seed, with as-run `drop_last` semantics. Verify `0<L<=U` and record actual values. All four conditions must execute exactly **U main joint optimizer iterations per epoch**, traverse the same canonical unlabeled stream once, and keep the original UCVME unsupervised branch active in **all U updates**, identically across conditions. UCVME has **two network optimizers**; both run exactly once in each joint iteration, not extra passes.

UCVME as-run main objective, per supervised-eligible step, has the structure:

```
loss_reg = heteroscedastic_labeled_loss_model0 + heteroscedastic_labeled_loss_model1
loss_cps = original_ucvme_unlabeled_cross_pseudo_supervision_and_variance_losses
loss_var_labeled = mean((var_model1_labeled - var_model0_labeled)^2)
loss_total = (loss_reg + loss_var_labeled) + w_ulb * loss_cps
```

The **supervision-controlled term MUST include BOTH** `loss_reg` and the *labeled-batch variance-consistency term* `loss_var_labeled`. The latter depends on labeled examples and must never be inadvertently active on an unlabeled-only step. Preserve the exact native definitions and weights within `loss_reg`, `loss_var_labeled` and `loss_cps`. Don't treat UCVME supervised loss as plain unweighted MSE, or move its labeled variance penalty into the unlabeled objective.

Define `S_t := loss_reg_t + loss_var_labeled_t` from the native labeled branch, and `C_t := w_ulb * loss_cps_t` from the native unlabeled branch (including all original uncertainty/pseudo-target/ensemble/variance terms). Implement the four conditions as `loss_t = alpha_t * S_t + C_t`, where:
- `front`: `alpha_t=1` at zero-based t=0,...,L-1; 0 otherwise; exactly one canonical labeled-loader traversal per epoch.
- `even`: `alpha_t=1` at `floor((j+0.5)*U/L)` for j=0,...,L-1, and 0 otherwise; exactly L uniquely spaced anchors.
- `back`: `alpha_t=1` at t=U-L,...,U-1; 0 otherwise; **pure** back-loading, no extra early anchor.
- `dense_mass_matched`: `alpha_t=L/U` on all U steps, recycling the labeled loader whenever exhausted; nominal `sum_t alpha_t=L`.

At alpha=0, **do not consume a labeled batch or compute a labeled forward/labeled variance loss**. Preserve complete unlabeled consistency computation and both optimizer updates. It is permissible that batchnorm/dropout RNG state differs across conditions because supervised forward passes differ; record the issue and (where feasible) isolate unlabeled data RNG to allow identical unlabeled order and stochastic ensemble sampling. Avoid hidden model-state leakage or unused labeled optimizer updates.

At alpha>0, use the identical labeled-loader stream/routing and supervised branch across conditions. Preserve native `samp_ssl` repeated stochastic predictions, stopgradient use, averaging `avg_mean01/avg_var01`, cross-model variance terms, stochastic dropout, and any method-specific uncertainty transformations. **No surrogate Plain objective or simplified pseudolabel loss**.

**Control interpretation:** FRONT/EVEN/BACK isolate temporal placement at matched U iterations, L active supervised positions, same labeled-example passes and same UCVME unlabeled objective (not necessarily identical gradient trajectories); DENSE matches only **nominal supervised coefficient mass**, not labeled-example exposure or realized gradient norm. A result favoring DENSE alone is not proof of pure timing in a fixed-exposure comparison.

**Important implementation detail:** The old UCVME benchmark may couple `run_epoch` training metric bookkeeping to availability of `yhat` and `y` from a labeled batch on every step; ensure sparse schedules with no labeled batch on inactive steps do not crash or silently fabricate metrics. Report supervised and unlabeled training losses, actual active anchors, valid data counts and epoch validation MAE; no dummy labels.

## Pairing, preflight and scientific controls

1. Check read-only original RSNA UCVME **as-run source** and its method-specific RSNA batch size and loss coefficients, sample transforms, optimizer/scheduler, evaluation aggregation (UCVME has two models), 30-epoch policy. Validate seed 0–2, 5% manifests, original split/cohort hashes, labeled-only target scaler, patient-disjoint split where applicable, original target unit months, no test leakage. If missing dataset/weights/venv on gpu05, report precise blockers; avoid silent substitutions.
2. In a new dedicated, scientifically coherent trainer, add only the supervised-anchor schedule factor. CPU/synthetic tests for exact 24 run keys and four `alpha_t` vectors, zero labeled consumption off-anchor, identical original UCVME unsupervised loss construction, all U main updates and two optimizer steps each, front/even/back labeled exposure equals one full labeled pass, dense coefficient mass L and unchanged method-specific loss. Assert scheduling index uniqueness for ratio-derived U,L.
3. Same seed/backbone across four conditions: **identical initial parameter hashes for both networks**, manifest/cohort, pretrained weights, optimizer setup, augmentation definitions, validation/test splits, and deterministic independent RNG streams for labeled loader, unlabeled shuffle, and (as feasible) UCVME stochastic pseudo-label ensemble. Verify no accidental perturbation of unlabeled sample order. Compare two network initial state digests and criterion hashes.
4. On **genuinely free A40 GPUs only 4,5,6,7**, run a brief **real-data FP32 preflight** for R50 and DINOv2 at the **native UCVME RSNA** batch size; check two-model VRAM, model output finite, uncertainty losses finite, backward and optimizer steps, step scheduling, validation, checkpoint write and reload. Do not silently change batch, precision, image size, optimizer, sampling count, pretrained weights or trainable parameters on OOM; mark blocked and report.
5. Save source SHA, versions, manifest digest, exact config, GPU UUID, all hyperparams and schedule contract assertions before full training.

## Run queue / result integrity

- Fresh root suggestion: `/nobackup/enzez/ucvme_rsna5_fixedbudget_24_worktree_20261119/artifacts/ucvme_rsna5_fixedbudget_24_20261119/` on gpu05 (only if path actually exists and sufficient quota; verify disk).
- Explicit `RUN_MATRIX.csv` with all **24 keys**; assign one trainer per safe GPU among IDs 4,5,6,7. Prefer load-balancing by backbone while respecting per-card VRAM, not assuming same speeds. Detached restartable controller+tmux (e.g. `ucvme-rsna5-24-20261119`) with file lock and atomic status. No overwrites/restarts of existing work, no duplicate or accidental extra jobs.
- **30 full epochs**, validation-selected best checkpoint (MAE **months**), checkpoint reload, exactly one final test evaluation after selection, store MAE/R² and ordered predictions privately. Use the original UCVME **two-model test aggregation** (e.g. averaging, if that is what the original actually used); confirm from run provenance, don't invent.
- Per run record all config/source/data/initial hashes, role RNG, exact U,L, optimizer steps and labeled-batch counts, nominal mass, best epoch, GPU UUID, 30 history rows, validation-only model selection, one-test-pass counter, finite predictions and metrics; persist exit codes and scientific completeness validation.
- Report `front-even`, `back-even`, `dense-even` paired seed-level MAE deltas per backbone, mean±population SD and exploratory 95% paired t-interval (n=3) only after all 3 paired seeds valid. DINO and R50 should be kept separate. No test-time tuning, post-hoc seed dropping or unverified claims of universal improvements.
- Local deliverables: `PREFLIGHT_REPORT.md`, `SOURCE_AND_LOSS_AUDIT.md`, `RUN_MATRIX.csv`, `QUEUE_STATUS.md`, `controller.log`, `README_MONITOR.md`, `aggregate_results.py`; use separate run dirs with `metrics.json`, `history.csv`, `config.json`, checkpoint, metadata. Keep artifact tree private on cluster.
- **If preflight passes, actually launch** the detached queue on observed idle cards and check real controller PID, trainer PID per used physical GPU, correct CUDA_VISIBLE_DEVICES, advancing finite losses. Report actual completed/running/queued/failed/blocked counts. If no eligible card or protocol mismatch, stop safely and give exact reason, rather than saying the campaign is launched.

## Codex response expected immediately

In Chinese or concise English: list host=instgpu-05.cs.wisc.edu, physical GPU IDs used from 4–7, verified model/VRAM and no interference, exact scientific config (not borrowed Plain recipe), total 24-run key breakdown and scheduler/controller status, real PIDs/tmux/log paths, and any discrepancies. Never claim 24 completed when only queued. Uploading this GitHub instruction file **does not** start Codex.
