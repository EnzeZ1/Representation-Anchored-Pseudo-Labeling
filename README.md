# Pseudo-label Regression Experiments

This repository studies semi-supervised regression with supervised, probe-filtered
(RAPL), and uniform pseudo-label baselines on UTKFace, IMDB-WIKI, and STS-B.

The canonical Heteroscedastic Pseudo-Labels (HPL) baseline is the official
third-party implementation in `Heteroscedastic-Pseudo-Labels-main/`. The former
project-owned HPL reimplementation has been retired and is no longer available
through the root `train.py` entry point.

## Repository structure

```text
train.py, uniform.py              Root CLI wrappers
preprocess_imdb_wiki.py           Root preprocessing CLI wrapper
models/                           ResNet-50, DINOv2, and DINOv3 model implementations
training/                         Training entry points and probe/RAPL method
data_processing/                  Dataset protocols, loaders, and preprocessing
scripts/                          Reproducible launch, queue, and reporting tools
baselines/                        Official-baseline adapters, patches, and revisions
analysis/                         Supported offline analyses
analysis/legacy_hpl/              Historical, unsupported project-owned HPL analyses
results/                          Untracked generated figures, logs, and diagnostics
data/                             Untracked local datasets
checkpoints/                      Untracked local model checkpoints
Heteroscedastic-Pseudo-Labels-main/
                                  Local official HPL checkout; not vendored
```

The three root CLI scripts are intentionally thin wrappers that preserve existing
commands. Active code must use package-qualified imports from `models`,
`training`, and `data_processing`.

The former inactive and nonfunctional `utkface_patched.py` module was removed and
remains recoverable through Git history.

Despite its historical name, `data_processing/hpl_data.py` is a shared
data-loading utility used by current probe, supervised, and uniform experiments
on the `utkface_official` split. It does not provide HPL training.

## Root-project methods

`train.py` supports:

- `--method probe`: probe-filtered/RAPL semi-supervised training.
- `--method supervised`: labeled-only training.

The uniform pseudo-label baseline is provided separately by `uniform.py`.
Root-project HPL training is intentionally unsupported.

### Training command

```bash
python train.py \
  -dataset {utkface,imdb_wiki,stsb,utkface_official} \
  --data_dir PATH \
  --method {probe,supervised} \
  --labeled_ratio 0.05 \
  --epochs 30 \
  --batch_size 32 \
  --save checkpoints/experiment.pt
```

Image experiments default to ResNet-50. DINOv2 can be selected with
`--backbone dinov2 --dino {s,b,l}`. Probe experiments can select a separate
frozen probe backbone with `--probe_backbone`.

### Uniform baseline command

```bash
python uniform.py \
  -dataset {utkface,imdb_wiki,stsb,utkface_official} \
  --data_dir PATH \
  --labeled_ratio 0.05 \
  --epochs 30 \
  --batch_size 32 \
  --save checkpoints/uniform.pt
```

Run either command from the repository root so local imports and relative paths
resolve consistently. Use `python train.py --help` and `python uniform.py --help`
for all options.

## Data and checkpoints

`data/` and `checkpoints/` are ignored by Git and are local artifacts. They may
contain large archives, extracted datasets, and historical checkpoints. Do not
assume another checkout contains them.

Legacy IMDB-WIKI preprocessing expects `metadata.json` in the supplied data
directory. Generate it from the repository root with:

```bash
python preprocess_imdb_wiki.py
```

Formal manifests under `data_processing/splits/` are generated local artifacts
and are not committed. Reproduce them with the protocol functions in
`data_processing/utkface_protocol.py` and
`data_processing/imdb_wiki_protocol.py`.

The canonical UTKFace cohort digest is
`61c397e0b6ac4be78db1b1c1431a65b031e2a2fd5089361e4e784ad21ad8af56`.
The formal benchmark uses seeds 0 through 5 and nested labeled subsets at
ratios 0.05, 0.10, and 0.20.

The official curated IMDB-WIKI-DIR metadata is `imdb_wiki.csv` from HPL
upstream commit `89f9f8bd467a0d3f81a8ada8708c3fe4fe31ca20`. Its expected SHA-256 is
`a31f1b43de6804ddbaa2316665a7364e74da3c5c497bdeafb40b910036f7f80b`.
The validated cohort has 191,509 training, 11,022 validation, and 11,022 test
records, with cohort digest
`919fe3e1b959e1fe75e08e83310a84c1c3a9d53a16812a1bb5f1e0117ba97f43`.
Place the images under `data/imdb_wiki/` and the official CSV in the pinned HPL
checkout; do not substitute a newly generated split.

The `utkface_official` loader currently reads split metadata and images from a
local official HPL checkout. This is a remaining runtime coupling; upstream
URLs, exact revisions, and reproducible patches are recorded in
`baselines/upstreams.yaml`.

## Official HPL baseline

Run official HPL experiments directly inside the corresponding third-party
dataset directory. Do not invoke them through root `train.py`.

UTKFace:

```bash
cd Heteroscedastic-Pseudo-Labels-main/utkface
conda env create -f environment.yml
conda activate hpl
python main_ours.py --data_dir PATH --output_dir PATH \
  --lr 1e-4 --fc_lr 1e-3 --unc_lr 1e-4 \
  --num_epochs 30 --batch_size 32
```

IMDB-WIKI:

```bash
cd Heteroscedastic-Pseudo-Labels-main/imdb_wiki
conda env create -f environment.yml
conda activate hpl
python main_ours.py --data_dir PATH --output_dir PATH \
  --lr 1e-4 --fc_lr 1e-3 --unc_lr 1e-4 \
  --num_epochs 30 --batch_size 48
```

STS-B:

```bash
cd Heteroscedastic-Pseudo-Labels-main/sts
conda env create -f environment.yml
conda activate sts
python main_ours.py --data_dir PATH --output_dir PATH \
  --labeled_ratio 0.1 --lr 1e-4 --fc_lr 1e-3 --unc_lr 1e-4 \
  --num_epochs 200 --batch_size 32
```

The third-party environments are authoritative for official HPL. Their exclusive
dependencies are deliberately not included in the root `requirements.txt`.
For a non-vendored checkout under `third_party/`, follow
`baselines/README.md`, check out the exact HPL revision from
`baselines/upstreams.yaml`, and apply `baselines/patches/hpl.patch`.

## Historical HPL analyses

Historical HPL checkpoints, logs, figures, result files, and diagnostic arrays
are retained. Comparison scripts that reconstruct the retired project-owned
uncertainty network are preserved for provenance under `analysis/legacy_hpl/`,
but are not part of the supported runnable analysis suite. See that directory's
README for details.

Those preserved scripts retain historical root-module imports that are no longer
provided. They are intentionally excluded from the active import and syntax
verification suite.

They have not been redirected to the official implementation because its model
and checkpoint formats differ, and doing so would change their scientific logic.
