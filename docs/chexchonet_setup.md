# CheXchoNet secure local setup

Protocol source code is public, but the licensed release is not. Never place release
metadata, image names, patient identifiers, manifests, predictions, or checkpoints in Git.

Place the authorized version `1.0.0` release under the user-only directory:

```
/nobackup/enzez/data/chexchonet/1.0.0/
├── metadata.csv   # official row-to-image mapping, patient key, LVIDd/IVSd/LVPWd
└── images/        # official image hierarchy referenced relative to metadata.csv
```

The official schema uses `patient_id`, `cxr_filename`, and lower-case target columns.
LVIDd is measured in centimeters. The formal LVIDd cohort includes finite values greater
than zero only. IVSd and LVPWd are preregistered future targets. Do not rename images,
synthesize missing records, or substitute another echocardiography dataset.

The fixed patient split seed is `20260801`, with one 90/5/5 patient partition shared by
every method and experimental seed. Protected generated manifests remain local and ignored.

After placement, run the aggregate-only audit and full decoder audit. Only a complete,
authorized release may be used to generate local patient-level manifests. The formal
launcher remains approval-gated after data readiness is established.
