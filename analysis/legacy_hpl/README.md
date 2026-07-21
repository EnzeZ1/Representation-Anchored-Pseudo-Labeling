# Legacy project-owned HPL analyses

The scripts in this directory depend on the retired project-owned
Heteroscedastic Pseudo-Labels (HPL) implementation. They are preserved only as
historical research artifacts and are not part of the supported runnable
analysis suite.

The retired implementation, formerly provided by the root `hpl.py` and
`dinov2_hpl.py` modules, remains available through this repository's Git
history. It must not be restored as the project's maintained HPL baseline merely
to run these scripts.

For supported HPL experiments, use the official third-party implementation and
its dataset-specific commands and environments under:

```text
Heteroscedastic-Pseudo-Labels-main/
```

The official implementation has different model and checkpoint assumptions, so
these historical scripts have not been silently redirected to it.
