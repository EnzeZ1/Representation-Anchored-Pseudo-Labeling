"""Frozen configuration and safety gates shared by CheXchoNet methods."""

from __future__ import annotations

from dataclasses import asdict, dataclass

HPL_UPSTREAM_COMMIT = "89f9f8bd467a0d3f81a8ada8708c3fe4fe31ca20"

@dataclass(frozen=True)
class FormalConfig:
    method: str
    backbone: str
    target: str = "lvidd"
    selection_metric: str = "lowest validation MAE in original target units"
    test_policy: str = "reload best checkpoint; exactly one final test inference"

    def metadata(self): return asdict(self)

def require_validated_manifest(manifest):
    if manifest.get("protocol_version") != "chexchonet-regression-v1":
        raise RuntimeError("Validated chexchonet-regression-v1 manifest required")
    if not manifest.get("manifest_sha256"):
        raise RuntimeError("Manifest checksum required")

def supervised_config(backbone): return FormalConfig("supervised_step_matched", backbone)
def rapl_config(backbone): return FormalConfig("rapl", backbone)
def hpl_config(backbone):
    value = FormalConfig("hpl", backbone).metadata(); value["official_hpl_upstream_commit"] = HPL_UPSTREAM_COMMIT
    return value
