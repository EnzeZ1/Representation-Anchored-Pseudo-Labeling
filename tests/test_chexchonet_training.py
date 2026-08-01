import unittest
from training.hpl_chexchonet import build_run_config as hpl
from training.rapl_chexchonet import build_run_config as rapl
from training.supervised_chexchonet import build_run_config as supervised

MANIFEST={"protocol_version":"chexchonet-regression-v1","manifest_sha256":"abc"}
class CheXchoNetTrainingTests(unittest.TestCase):
    def test_configs_are_method_explicit(self):
        self.assertEqual(supervised("resnet50",MANIFEST)["method"],"supervised_step_matched")
        self.assertTrue(rapl("dinov2_vits14",MANIFEST)["anchor_frozen"])
        self.assertTrue(hpl("resnet50",MANIFEST)["bilevel_meta_optimization"])
    def test_unvalidated_manifest_rejected(self):
        with self.assertRaises(RuntimeError):supervised("resnet50",{})
if __name__=="__main__":unittest.main()
