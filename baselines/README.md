# Official baseline adapters

`upstreams.yaml` pins the pristine upstream revisions used by the controlled
UTKFace benchmark. Clone each repository at its recorded commit under
`third_party/`, then apply the corresponding file in `patches/` from the
upstream repository root. The full upstream repositories and installed runtime
dependencies remain ignored and are not vendored into this repository.

These runs use official algorithm implementations with a shared dataset,
manifest, augmentation, label-scaling, validation-selection, and final-test
protocol. They are not claims of exact reproduction of the papers' original
environments or numerical results.

Recreate a checkout with, for example:

```bash
git clone https://github.com/xmed-lab/UCVME.git third_party/UCVME
git -C third_party/UCVME checkout 9f27a579cc8cb8806a56b52c3f888b647d22a074
git -C third_party/UCVME apply ../../baselines/patches/ucvme.patch
```

Use the analogous URL, commit, directory, and patch recorded in
`upstreams.yaml` for the other methods. Generated environments and run metadata
belong under ignored `artifacts/` directories and are not part of the Git
snapshot. Each run records Python, PyTorch, torchvision, Pillow, and NumPy
versions in its metadata.

For HPL:

```bash
git clone https://github.com/sxq11/Heteroscedastic-Pseudo-Labels.git \
  third_party/Heteroscedastic-Pseudo-Labels
git -C third_party/Heteroscedastic-Pseudo-Labels checkout \
  89f9f8bd467a0d3f81a8ada8708c3fe4fe31ca20
git -C third_party/Heteroscedastic-Pseudo-Labels apply \
  ../../baselines/patches/hpl.patch
```

Formal manifests are intentionally not committed because cohort files contain
the complete local dataset index and are large. Generate them deterministically
with `data_processing/utkface_protocol.py` or
`data_processing/imdb_wiki_protocol.py`; expected dataset digests and official
metadata hashes are documented in the root README.
