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
models/                           ResNet-50 and DINOv2 model implementations
training/                         Training entry points and probe/RAPL method
data_processing/                  Dataset loaders and IMDB-WIKI preprocessing
analysis/                        Supported offline analyses
analysis/legacy_hpl/             Historical, unsupported project-owned HPL analyses
results/                         Tracked figures, logs, and diagnostics
data/                            Untracked local datasets
checkpoints/                     Untracked local model checkpoints
Heteroscedastic-Pseudo-Labels-main/
                                 Unmodified official third-party HPL project
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

IMDB-WIKI experiments expect `metadata.json` in the supplied data directory.
Generate it from the repository root with:

```bash
python preprocess_imdb_wiki.py
```

The `utkface_official` loader currently reads split metadata and images from
`Heteroscedastic-Pseudo-Labels-main/utkface/data/`. This is a remaining coupling
between the root project and the local third-party checkout.

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
