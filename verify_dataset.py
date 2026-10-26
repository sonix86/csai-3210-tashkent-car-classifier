"""Validate all active annotations and originals without starting any model."""
import csv
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageOps
import torch
import project


def main():
    root = Path(__file__).resolve().parent
    rows = project.read_rows(root / "data")
    dataset = project.CarDataset(rows, root / "data", sorted({r["brand"] for r in rows}))
    for i, row in enumerate(rows):
        with Image.open(root / row["photo"]) as im:
            width, height = ImageOps.exif_transpose(im).size
        left, top, right, bottom = json.loads(row["bbox"])
        assert 0 <= left < right <= width and 0 <= top < bottom <= height
        with Image.open(root / "data" / row["crop"]) as crop:
            assert crop.size == (right-left, bottom-top)
            crop.verify()
        tensor, _, _ = dataset[i]
        assert tensor.shape == (3, 224, 224) and torch.isfinite(tensor).all()
    baseline = json.loads((root / 'data/original_sha256.json').read_text(encoding='utf8'))
    assert len(baseline) == 76
    assert all(hashlib.sha256((root/'cars_photos'/name).read_bytes()).hexdigest().lower() == digest.lower() for name,digest in baseline.items())
    train, val = project.split_indices(rows)
    report = {"active_crops": len(rows), "verified_originals_unchanged": len(baseline),
        "vehicle_groups": len({r["vehicle_id"] for r in rows}), "split_groups": len({r["source"] for r in rows}),
        "train_crops_preview": len(train), "validation_crops_preview": len(val),
        "validation_only_brands": sorted({rows[i]["brand"] for i in val} - {rows[i]["brand"] for i in train}),
        "brands_without_validation": sorted({rows[i]["brand"] for i in train} - {rows[i]["brand"] for i in val}),
        "verification_did_not_run_training_or_accuracy_evaluation": True}
    (root / 'checks').mkdir(exist_ok=True)
    (root / "checks" / "dataset_verification.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
