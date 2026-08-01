"""CheXchoNet adapter retaining official-compatible HPL provenance."""
from training.chexchonet_common import hpl_config, require_validated_manifest

def build_run_config(backbone, manifest):
    require_validated_manifest(manifest)
    value = hpl_config(backbone)
    value.update({"uncertainty_learner": True, "heteroscedastic_weighting": True,
                  "bilevel_meta_optimization": True})
    return value
