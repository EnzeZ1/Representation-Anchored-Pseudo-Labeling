"""Pure supervised CheXchoNet adapter configuration.

Training launch remains intentionally gated on protected local manifests.
"""
from training.chexchonet_common import supervised_config, require_validated_manifest

def build_run_config(backbone, manifest):
    require_validated_manifest(manifest)
    return supervised_config(backbone).metadata()
