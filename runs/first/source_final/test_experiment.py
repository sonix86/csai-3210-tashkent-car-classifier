"""Tests for honest evaluation, provenance and immutable prediction inputs."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from experiment import check_split, fingerprint, head_metrics
from inference import protect_original, save_annotated


class ExperimentTests(unittest.TestCase):
    def test_absent_classes_are_unvalidated_not_successes(self):
        metrics = head_metrics([0, 1, 1], [0, 2, 2], ["a", "b", "c"])
        self.assertEqual(metrics["correct"], 1)
        self.assertEqual(metrics["unvalidated_classes"], ["c"])
        self.assertIsNone(metrics["per_class"]["c"]["recall"])
        self.assertEqual(metrics["per_class"]["b"]["recall"], 0)
        with self.assertRaises(ValueError):
            head_metrics([], [], ["a"])

    def test_fingerprint_detects_crop_and_label_changes(self):
        with tempfile.TemporaryDirectory(dir="checks") as temp:
            path = Path(temp)/"crop.jpg"
            path.write_bytes(b"one")
            rows = [{"crop": "crop.jpg", "brand": "Kia"}]
            digest = fingerprint(rows, temp)
            path.write_bytes(b"two")
            self.assertNotEqual(digest, fingerprint(rows, temp))
            path.write_bytes(b"one")
            self.assertNotEqual(digest, fingerprint([dict(rows[0], brand="BMW")], temp))

    def test_shared_vehicle_or_photo_is_rejected(self):
        for shared in ("vehicle_id", "photo", "source"):
            a = {"brand": "Kia", "source": "s1", "photo": "p1", "vehicle_id": "v1"}
            b = {"brand": "Kia", "source": "s2", "photo": "p2", "vehicle_id": "v2"}
            b[shared] = a[shared]
            with self.assertRaisesRegex(ValueError, "leakage"):
                check_split([a, b], [0], [1])

    def test_output_protection_and_annotation_copy(self):
        with self.assertRaises(ValueError):
            protect_original(Path(__file__).parent/"cars_photos/any.jpg")
        with tempfile.TemporaryDirectory(dir="checks") as temp:
            frame = np.zeros((60, 60, 3), dtype=np.uint8)
            before = frame.copy()
            save_annotated(frame, [{"box": [5, 5, 40, 40], "index": 1, "brand": "Kia", "colour": "white"}], Path(temp)/"output.jpg")
            np.testing.assert_array_equal(frame, before)


if __name__ == "__main__":
    unittest.main()
