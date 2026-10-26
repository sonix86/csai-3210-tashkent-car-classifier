"""CSAI-3210: original dual-head brand/colour classifier with COCO detection."""
import argparse
import csv
import json
import os
import uuid
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from dataset_utils import FIELDS, assign_split_groups, grouped_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

COLOURS = ["white", "black", "grey", "red", "blue", "green", "brown", "yellow"]


def read_image(path):
    with Image.open(path) as img:
        return cv2.cvtColor(np.array(ImageOps.exif_transpose(img).convert("RGB")), cv2.COLOR_RGB2BGR)


def write_labels(path, rows):
    temp = path.with_suffix(".csv.tmp")
    with temp.open("w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(assign_split_groups(rows))
    temp.replace(path)


def label(args):
    photos = Path(args.photos)
    data = Path(args.data)
    crops = data / "crops"
    crops.mkdir(parents=True, exist_ok=True)
    csv_path = data / "labels.csv"
    seen = set()
    rows = []
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != FIELDS:
                raise ValueError("Legacy annotation schema: migrate to explicit photo and vehicle_id before appending.")
            rows = list(reader)
            seen = {r["photo"] for r in rows}
    files = sorted(p for p in photos.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not files:
        raise SystemExit("No JPG/PNG images found. Add real phone photos in photos/ first.")
    try:
        for path in files:
            photo = Path(os.path.relpath(path.resolve(), data.resolve().parent)).as_posix()
            if photo in seen and not args.revisit:
                print("Skipping already annotated photo:", path)
                continue
            try:
                frame = read_image(path)
            except (OSError, ValueError):
                print("Unreadable image:", path)
                continue
            print("\nPHOTO", path, "— select car ROI; cancel to finish")
            count = 0
            while True:
                box = cv2.selectROI("Select car, Enter to confirm, c/Esc to stop", frame, fromCenter=False, showCrosshair=True)
                x, y, w, h = map(int, box)
                if w < 20 or h < 20:
                    break
                brand = input("Badge brand (blank to skip): ").strip()
                if not brand:
                    continue
                colour = input("Colour [" + ", ".join(COLOURS) + "]: ").strip().lower()
                if colour not in COLOURS:
                    print("Invalid colour; skipped")
                    continue
                print("Existing IDs for this brand:", sorted({r["vehicle_id"] for r in rows if r["brand"] == brand}))
                vehicle_id = input("Physical vehicle ID (reuse for another view; required): ").strip()
                if not vehicle_id:
                    print("Missing vehicle ID; skipped")
                    continue
                if any(r["vehicle_id"] == vehicle_id and (r["brand"], r["colour"]) != (brand, colour) for r in rows):
                    print("Conflicting brand/colour for this vehicle ID; skipped")
                    continue
                crop = frame[y:y + h, x:x + w]
                bbox = json.dumps([x, y, x+w, y+h])
                if any(r["photo"] == photo and r["bbox"] == bbox for r in rows):
                    print("This box is already annotated; skipped")
                    continue
                crop_name = f"manual_{uuid.uuid4().hex}.jpg"
                Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).save(crops / crop_name, quality=95)
                rows.append({"crop": f"crops/{crop_name}", "source": "", "brand": brand,
                    "colour": colour, "vehicle_id": vehicle_id, "photo": photo,
                    "bbox": bbox, "evidence": "Interactive manual annotation"})
                write_labels(csv_path, rows)
                count += 1
            cv2.destroyAllWindows()
            print(f"Saved {count} annotated cars")
    finally:
        cv2.destroyAllWindows()


def read_rows(data):
    with (Path(data) / "labels.csv").open(newline="", encoding="utf8") as f:
        rows = list(csv.DictReader(f))
    seen_crops = set()
    vehicle_labels = {}
    for line, row in enumerate(rows, 2):
        if not all(row.get(k) for k in ("crop", "source", "brand", "colour")):
            raise ValueError(f"labels.csv:{line}: missing required value")
        if row["colour"] not in COLOURS or not (Path(data) / row["crop"]).is_file():
            raise ValueError(f"labels.csv:{line}: invalid colour or missing crop")
        if row["crop"] in seen_crops:
            raise ValueError(f"labels.csv:{line}: duplicate crop")
        seen_crops.add(row["crop"])
        if "vehicle_id" in row:
            if not row["vehicle_id"] or not row.get("photo") or not row.get("bbox"):
                raise ValueError(f"labels.csv:{line}: missing provenance")
            vehicle = row["vehicle_id"]
            pair = row["brand"], row["colour"]
            if vehicle in vehicle_labels and vehicle_labels[vehicle] != pair:
                raise ValueError(f"Conflicting labels for vehicle {vehicle}")
            vehicle_labels[vehicle] = pair
    if rows and "vehicle_id" in rows[0]:
        expected = assign_split_groups(rows)
        if any(a["source"] != b["source"] for a, b in zip(rows, expected)):
            raise ValueError("Stale split groups; rebuild groups from photo and vehicle_id")
    if len(rows) < 10:
        raise ValueError("Need at least 10 labelled car crops; collect more data first.")
    return rows


def split_indices(rows, seed=42):
    return grouped_split(rows, seed)


def image_transform(training):
    ops = [transforms.Resize((224, 224))]
    if training:
        ops += [transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(8),
                transforms.ColorJitter(brightness=.2, contrast=.2, saturation=.15)]
    ops += [transforms.ToTensor(), transforms.Normalize([.485, .456, .406], [.229, .224, .225])]
    return transforms.Compose(ops)


class CarDataset(Dataset):
    def __init__(self, rows, data, brands, training=False):
        self.rows, self.data = rows, Path(data)
        self.brand_to_id = {b: i for i, b in enumerate(brands)}
        self.transform = image_transform(training)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.data / row["crop"]) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            tensor = self.transform(img)
        return tensor, self.brand_to_id[row["brand"]], COLOURS.index(row["colour"])


class DualHeadRegNet(nn.Module):
    """Shared RegNet-Y-800MF features with independent brand and colour heads."""
    def __init__(self, n_brands, pretrained=False):
        super().__init__()
        weights = models.RegNet_Y_800MF_Weights.DEFAULT if pretrained else None
        self.body = models.regnet_y_800mf(weights=weights)
        width = self.body.fc.in_features
        self.body.fc = nn.Identity()
        self.brand_head = nn.Sequential(nn.Dropout(.25), nn.Linear(width, 256),
                                        nn.ReLU(), nn.Linear(256, n_brands))
        self.colour_head = nn.Sequential(nn.Dropout(.1), nn.Linear(width, 128),
                                         nn.ReLU(), nn.Linear(128, len(COLOURS)))

    def forward(self, x):
        features = self.body(x)
        return self.brand_head(features), self.colour_head(features)


def predict_batch(model, loader, device):
    model.eval()
    brand_truth, brand_pred, colour_truth, colour_pred = [], [], [], []
    with torch.inference_mode():
        for x, b, c in loader:
            bp, cp = model(x.to(device))
            brand_truth.extend(b.tolist())
            brand_pred.extend(bp.argmax(1).cpu().tolist())
            colour_truth.extend(c.tolist())
            colour_pred.extend(cp.argmax(1).cpu().tolist())
    return brand_truth, brand_pred, colour_truth, colour_pred


def train(args):
    from experiment import train as run_train
    return run_train(args)


def load_model(path, device, return_checkpoint=False):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("colours") != COLOURS:
        raise ValueError("Checkpoint colour order differs from current schema")
    brands = checkpoint["brands"]
    model = DualHeadRegNet(len(brands)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return (model, brands, checkpoint) if return_checkpoint else (model, brands)


def evaluate(args):
    from experiment import evaluate as run_evaluate
    return run_evaluate(args)


def predict(args):
    from inference import predict as run_predict
    return run_predict(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("label")
    p.add_argument("--photos", default="photos")
    p.add_argument("--data", default="data")
    p.add_argument("--revisit", action="store_true", help="Add more boxes to an already annotated photo")
    p = sub.add_parser("train")
    p.add_argument("--data", default="data")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch", type=int, default=24)
    p.add_argument("--lr", type=float, default=0.0001)
    p.add_argument("--head-lr", type=float, default=0.0005)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="runs/first")
    p = sub.add_parser("evaluate")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--data", default="data")
    p.add_argument("--weights", required=True)
    p.add_argument("--out", default="runs/first")
    p = sub.add_parser("predict")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--json", help="Write machine-readable predictions and timings")
    p.add_argument("--image", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--detector", default="yolo11n.pt")
    p.add_argument("--save", default="result.jpg")
    p.add_argument("--detect-conf", type=float, default=.25)
    p.add_argument("--brand-threshold", type=float, default=.4)
    p.add_argument("--min-size", type=int, default=35)
    args = parser.parse_args()
    {"label": label, "train": train, "evaluate": evaluate, "predict": predict}[args.command](args)


if __name__ == "__main__":
    main()
