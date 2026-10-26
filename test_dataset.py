"""Regression checks for annotation persistence and vehicle/photo leakage."""
import argparse
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image
import project
from dataset_utils import FIELDS, assign_split_groups


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Path('checks').mkdir(exist_ok=True)

    def test_transitive_photo_vehicle_groups(self):
        rows = [dict(vehicle_id="a", photo="one"), dict(vehicle_id="b", photo="one"),
                dict(vehicle_id="b", photo="two"), dict(vehicle_id="c", photo="three")]
        grouped = assign_split_groups(rows)
        self.assertEqual(len({r["source"] for r in grouped[:3]}), 1)
        self.assertNotEqual(grouped[0]["source"], grouped[3]["source"])

    def test_real_splits_have_no_leakage_and_no_unseen_validation_brands(self):
        rows = project.read_rows("data")
        for seed in range(100):
            train, val = project.split_indices(rows, seed)
            self.assertEqual(sorted(train + val), list(range(len(rows))))
            for key in ("source", "vehicle_id", "photo"):
                self.assertFalse({rows[i][key] for i in train} & {rows[i][key] for i in val})
            self.assertLessEqual({rows[i]["brand"] for i in val}, {rows[i]["brand"] for i in train})
            # Current data has six brands with independently holdable groups.
            self.assertLessEqual({"BMW", "BYD", "Chevrolet", "Deepal", "Kia", "Leapmotor"},
                                 {rows[i]["brand"] for i in val})
            self.assertLessEqual({"white", "black", "grey"}, {rows[i]["colour"] for i in val})
        self.assertEqual(project.split_indices(rows, 42), project.split_indices(rows, 42))

    def test_unicode_label_and_resume_preserves_identity(self):
        with tempfile.TemporaryDirectory(dir="checks") as temp:
            root = Path(temp)
            photos, data = root / "фото", root / "data"
            photos.mkdir()
            first = photos / "машина.jpg"
            Image.new("RGB", (100, 100), "white").save(first)
            args = argparse.Namespace(photos=photos, data=data, revisit=False)
            def annotate():
                with patch.object(project.cv2, "selectROI", side_effect=[(10, 10, 60, 60), (0, 0, 0, 0)]), \
                     patch.object(project.cv2, "destroyAllWindows"), \
                     patch("builtins.input", side_effect=["Kia", "white", "same_car"]):
                    project.label(args)
            annotate()
            with patch.object(project.cv2, "selectROI", side_effect=AssertionError("Should skip existing photo")):
                project.label(args)
            Image.new("RGB", (100, 100), "white").save(photos / "ракурс.jpg")
            annotate()
            with (data / "labels.csv").open(encoding="utf8", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 2)
            self.assertEqual(len({r["crop"] for r in rows}), 2)
            self.assertEqual(len({r["source"] for r in rows}), 1)
            self.assertEqual(json.loads(rows[0]["bbox"]), [10, 10, 70, 70])
            for row in rows:
                with Image.open(data / row["crop"]) as im:
                    self.assertEqual(im.size, (60, 60))

    def test_bad_annotation_is_not_silently_dropped(self):
        with tempfile.TemporaryDirectory(dir="checks") as temp:
            data = Path(temp)
            with (data / "labels.csv").open("w", newline="", encoding="utf8") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerow(dict(crop="missing.jpg", source="a", brand="Kia", colour="invalid"))
            with self.assertRaisesRegex(ValueError, "invalid colour or missing crop"):
                project.read_rows(data)


if __name__ == "__main__":
    unittest.main()

