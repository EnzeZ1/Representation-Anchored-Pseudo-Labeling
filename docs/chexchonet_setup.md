# CheXchoNet secure local setup

Protocol source code is public, but the licensed release is not. Never place release
metadata, image names, patient identifiers, manifests, predictions, or checkpoints in Git.

Place the authorized version `1.0.0` release under the user-only directory:

```
/nobackup/enzez/data/chexchonet/1.0.0/
├── metadata.csv   # official row-to-image mapping, patient key, LVIDd/IVSd/LVPWd
└── images/        # official image hierarchy referenced relative to metadata.csv
```

The metadata filename may retain its official release name, but exactly one top-level CSV
must identify image paths, patient membership, and LVIDd. IVSd and LVPWd are preregistered
future targets. Do not rename images, synthesize missing records, or substitute another
echocardiography dataset.

After placement, run the aggregate-only audit and full decoder audit. Only a complete,
authorized release may be used to generate local patient-level manifests. The formal
launcher remains approval-gated after data readiness is established.
