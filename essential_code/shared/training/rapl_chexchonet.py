"""CheXchoNet RAPL adapter using the existing visual RAPL semantics."""
from training.chexchonet_common import rapl_config, require_validated_manifest

def build_run_config(backbone, manifest):
    require_validated_manifest(manifest)
    value = rapl_config(backbone).metadata()
    value.update({"tau": 1.0, "trust_formula": "1/(1+abs(target_pseudo-frozen_probe))",
                  "probe_fit_data": "labeled subset only", "anchor_frozen": True})
    return value
