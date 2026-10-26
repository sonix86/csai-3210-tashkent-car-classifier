"""Build reviewed crops from explicit annotations; never alter source photos."""
import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from PIL import Image, ImageOps
from dataset_utils import FIELDS, assign_split_groups, grouped_split

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "data")
    parser.add_argument("--replace", action="store_true", help="Back up existing labels before replacing the manifest")
    args = parser.parse_args()
    out = args.out.resolve()
    if (out / "labels.csv").exists() and not args.replace:
        parser.error("labels.csv already exists. Use a new --out, or --replace to back it up first.")
    with (ROOT / "reviewed_annotations.csv").open(newline="", encoding="utf8") as f:
        annotations = list(csv.DictReader(f))
    originals = sorted((ROOT / "cars_photos").glob("*.jpg"))
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in originals}
    manifest = []
    for a in annotations:
        photo = f"cars_photos/photo_{int(a['photo_number'])}_2026-10-26_17-10-50.jpg"
        coords = [float(a[k]) for k in ("left", "top", "right", "bottom")]
        assert 0 <= coords[0] < coords[2] <= 1 and 0 <= coords[1] < coords[3] <= 1
        with Image.open(ROOT / photo) as im:
            width, height = ImageOps.exif_transpose(im).size
        bbox = [int(coords[0]*width), int(coords[1]*height), int(coords[2]*width), int(coords[3]*height)]
        identity = json.dumps([photo, bbox, a["vehicle_id"]], sort_keys=True)
        crop = "crops/reviewed_v2_" + hashlib.sha256(identity.encode()).hexdigest()[:16] + ".jpg"
        manifest.append(dict(crop=crop, source="", brand=a["brand"], colour=a["colour"],
            vehicle_id=a["vehicle_id"], photo=photo, bbox=json.dumps(bbox), evidence=a["evidence"]))
    manifest = assign_split_groups(manifest)
    if len({r["crop"] for r in manifest}) != len(manifest):
        raise ValueError("Duplicate annotation")
    for vehicle in {r["vehicle_id"] for r in manifest}:
        assert len({(r["brand"], r["colour"]) for r in manifest if r["vehicle_id"] == vehicle}) == 1
    # Check that a valid split is possible before modifying output files.
    train, val = grouped_split(manifest)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "labels.csv").exists():
        backup = out / "backups" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup.mkdir(parents=True)
        for name in ("labels.csv", "review_summary.json", "split_preview.json"):
            if (out / name).exists():
                shutil.copy2(out / name, backup / name)
    (out / "crops").mkdir(exist_ok=True)
    for row in manifest:
        target = out / row["crop"]
        if not target.exists():
            with Image.open(ROOT / row["photo"]) as im:
                ImageOps.exif_transpose(im).convert("RGB").crop(json.loads(row["bbox"])).save(target, quality=95)
        with Image.open(target) as im:
            im.verify()
    temp = out / "labels.csv.tmp"
    with temp.open("w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(manifest)
    temp.replace(out / "labels.csv")
    colours = ["white", "black", "grey", "red", "blue", "green", "brown", "yellow"]
    vehicles = {r["vehicle_id"]: r for r in manifest}
    summary = {"reviewed_crops": len(manifest), "unique_vehicle_groups": len(vehicles),
        "split_groups": len({r["source"] for r in manifest}),
        "photos_with_labels": len({r["photo"] for r in manifest}), "photos_reviewed": len(originals),
        "brands": dict(Counter(r["brand"] for r in manifest)),
        "colours": dict(Counter(r["colour"] for r in manifest)),
        "status": "VISUALLY REVIEWED; MODEL RESULTS ARE STORED SEPARATELY IN runs/"}
    (out / "review_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    for filename, items in (("counts_crops.csv", manifest), ("counts_vehicles.csv", list(vehicles.values()))):
        with (out / filename).open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["brand"] + colours + ["total"])
            for brand in sorted({r["brand"] for r in items}):
                counts = Counter(r["colour"] for r in items if r["brand"] == brand)
                writer.writerow([brand] + [counts[c] for c in colours] + [sum(counts.values())])
            counts = Counter(r["colour"] for r in items)
            writer.writerow(["TOTAL"] + [counts[c] for c in colours] + [len(items)])
    preview = {"seed": 42, "train_crops": [manifest[i]["crop"] for i in train],
        "val_crops": [manifest[i]["crop"] for i in val],
        "train_sources": sorted({manifest[i]["source"] for i in train}),
        "val_sources": sorted({manifest[i]["source"] for i in val}),
        "brands_without_validation": sorted({r["brand"] for r in manifest} - {manifest[i]["brand"] for i in val}),
        "note": "Preview only, no training performed. Singleton groups stay in train."}
    (out / "split_preview.json").write_text(json.dumps(preview, indent=2), encoding="utf8")
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in originals}
    assert before == after, "Original photos changed"
    (out / "original_sha256.json").write_text(json.dumps(after, indent=2), encoding="utf8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
