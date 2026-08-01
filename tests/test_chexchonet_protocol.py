import tempfile, unittest
from pathlib import Path
from PIL import Image
from data_processing.chexchonet import audit_release, load_records
from data_processing.chexchonet_protocol import RATIOS, build_train_transform, build_strong_transform, build_evaluation_transform, generate_patient_manifest, validate_nested

class CheXchoNetProtocolTests(unittest.TestCase):
    def records(self):
        return [{"patient_id":f"p{i:03d}","image_path":f"x{i}.png","targets":{"lvidd":float(i%7+1),"ivsd":None,"lvpwd":None}} for i in range(1000)]
    def test_patient_splits_and_nesting(self):
        records=self.records(); manifests={r:generate_patient_manifest(records,seed=2,ratio=r) for r in RATIOS}; validate_nested(manifests)
        for manifest in manifests.values():
            groups=[set(manifest["labeled_patients"]),set(manifest["unlabeled_patients"]),set(manifest["patient_splits"]["validation"]),set(manifest["patient_splits"]["test"])]
            self.assertFalse(any(groups[i]&groups[j] for i in range(4) for j in range(i+1,4)))
    def test_transform_has_no_horizontal_flip(self):
        for transform in (build_train_transform(),build_strong_transform(),build_evaluation_transform()):
            text=repr(transform);self.assertNotIn("Crop",text);self.assertNotIn("Flip",text)
        self.assertIn("degrees=[-2.0, 2.0]",repr(build_train_transform()))
        self.assertIn("degrees=[-5.0, 5.0]",repr(build_strong_transform()))
    def test_secure_release_audit(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/"images").mkdir();Image.new("RGB",(4,4)).save(root/"images/a.png")
            (root/"metadata.csv").write_text("patient_id,cxr_filename,lvidd,ivsd,lvpwd\np1,a.png,4.2,1.0,1.1\n")
            audit=audit_release(root,decode=True);self.assertTrue(audit.ready);self.assertEqual(audit.unique_patients,1)
    def test_invalid_lvidd_excluded_not_unlabeled(self):
        records=self.records();records[0]["targets"]["lvidd"]=0;records[1]["targets"]["lvidd"]=float("nan")
        manifest=generate_patient_manifest(records,seed=0,ratio=.05)
        all_indices=set().union(*(set(manifest["indices"][x]) for x in ("labeled","unlabeled","validation","test")))
        self.assertNotIn(0,all_indices);self.assertNotIn(1,all_indices)
    def test_ambiguous_metadata_fails(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/"a.csv").touch();(root/"b.csv").touch()
            with self.assertRaisesRegex(ValueError,"one official metadata"):audit_release(root)
if __name__=="__main__":unittest.main()
