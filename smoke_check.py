"""Read-only dataset audit and GPU smoke check; no optimizer or accuracy metrics."""
import csv
import argparse
import contextlib
import hashlib
import io
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".cache" / "ultralytics"))

import cv2
import torch
import torchvision
from PIL import Image, ImageDraw, ImageOps
from torch.utils.data import DataLoader
import project


def main():
    out = ROOT / "checks"
    out.mkdir(exist_ok=True)
    report = {"python": __import__("sys").version, "torch": torch.__version__,
              "torchvision": torchvision.__version__, "cuda_runtime": torch.version.cuda,
              "cuda_available": torch.cuda.is_available()}
    originals = sorted((ROOT / "cars_photos").glob("*.jpg"))
    rows = project.read_rows(ROOT / "data")
    crops = [ROOT / "data" / r["crop"] for r in rows]
    hashes = defaultdict(list)
    for path in originals + crops:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            img.load()
        hashes[hashlib.sha256(path.read_bytes()).hexdigest()].append(str(path.relative_to(ROOT)))
    report["originals"] = len(originals)
    report["crops"] = len(crops)
    report["exact_duplicates"] = [v for v in hashes.values() if len(v) > 1]
    baseline = json.loads((ROOT / 'data/original_sha256.json').read_text(encoding='utf8'))
    assert all(hashlib.sha256((ROOT/'cars_photos'/name).read_bytes()).hexdigest().lower() == digest.lower() for name,digest in baseline.items())
    report["original_hashes_unchanged"] = True
    for offset in range(0, len(originals), 24):
        sheet = Image.new("RGB", (1200, 4 * 190), "white")
        draw = ImageDraw.Draw(sheet)
        for j, path in enumerate(originals[offset:offset + 24]):
            with Image.open(path) as img:
                thumb = ImageOps.exif_transpose(img).convert("RGB")
                thumb.thumbnail((196, 160))
            x, y = (j % 6) * 200, (j // 6) * 190
            sheet.paste(thumb, (x, y))
            draw.text((x, y + 163), path.stem.split("_2026")[0], fill="black")
        sheet.save(out / f"contact_{offset // 24 + 1}.jpg")
    rows = project.read_rows(ROOT / "data")
    brands = sorted({r["brand"] for r in rows})
    train, val = project.split_indices(rows)
    report["labels"] = len(rows)
    report["groups"] = len({r.get("vehicle_id", r["source"]) for r in rows})
    report["split_groups"] = len({r["source"] for r in rows})
    report["brand_counts"] = dict(Counter(r["brand"] for r in rows))
    report["colour_counts"] = dict(Counter(r["colour"] for r in rows))
    report["brand_groups"] = {b: len({r.get("vehicle_id", r["source"]) for r in rows if r["brand"] == b}) for b in brands}
    report["split"] = {"train": len(train), "validation": len(val),
        "validation_only_brands": sorted({rows[i]["brand"] for i in val} - {rows[i]["brand"] for i in train}),
        "train_only_brands": sorted({rows[i]["brand"] for i in train} - {rows[i]["brand"] for i in val})}
    assert not ({rows[i]["source"] for i in train} & {rows[i]["source"] for i in val})
    dataset = project.CarDataset(rows, ROOT / "data", brands)
    for i in range(len(dataset)):
        x, b, c = dataset[i]
        assert x.shape == (3, 224, 224) and torch.isfinite(x).all()
    report["opencv_unicode_read"] = cv2.imread(str(originals[0])) is not None
    assert torch.cuda.is_available(), "CUDA unavailable"
    report["gpu"] = torch.cuda.get_device_name(0)
    report["gpu_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
    report["capability"] = torch.cuda.get_device_capability(0)
    torch.manual_seed(42)
    model = project.DualHeadRegNet(len(brands), pretrained=True).cuda()
    x, b, c = next(iter(DataLoader(dataset, batch_size=2)))
    model.train()
    bp, cp = model(x.cuda())
    loss = .6 * torch.nn.functional.cross_entropy(bp, b.cuda()) + .4 * torch.nn.functional.cross_entropy(cp, c.cuda())
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    model.zero_grad(set_to_none=True)
    model.eval()
    with torch.inference_mode():
        bp, cp = model(x.cuda())
    assert bp.shape == (2, len(brands)) and cp.shape == (2, 8)
    with tempfile.TemporaryDirectory(dir=out) as temp:
        checkpoint = Path(temp) / "smoke_only.pt"
        torch.save({"model": model.state_dict(), "brands": brands,
                    "colours": project.COLOURS}, checkpoint)
        restored, restored_brands = project.load_model(checkpoint, "cuda")
        with torch.inference_mode():
            rb, rc = restored(x.cuda())
        assert restored_brands == brands and torch.allclose(bp, rb) and torch.allclose(cp, rc)
        del restored
        from ultralytics import YOLO
        detector_path = ROOT / "yolo11n.pt"
        detector = YOLO(str(detector_path))
        with Image.open(originals[0]) as img:
            source = __import__("numpy").array(ImageOps.exif_transpose(img).convert("RGB"))[:, :, ::-1].copy()
        detected = detector.predict(source=source, device=0, verbose=False)[0]
        report["detector_gpu"] = "passed"
        report["detector_car_boxes_on_sample"] = sum(int(b.cls.item()) == 2 for b in detected.boxes)
        del detector
        # Exercise the real CLI handler without presenting random head labels as results.
        with contextlib.redirect_stdout(io.StringIO()):
            project.predict(argparse.Namespace(image=str(originals[0]), weights=str(checkpoint),
                detector=str(detector_path), save=str(Path(temp) / "pipeline.jpg"),
                detect_conf=.25, brand_threshold=.4, min_size=35))
        assert (Path(temp) / "pipeline.jpg").is_file()
        report["predict_pipeline"] = "passed; untrained heads, temporary output deleted, no quality evaluation"
    report["checkpoint_roundtrip"] = "passed; temporary checkpoint deleted"
    torch.cuda.synchronize()
    report["gpu_forward_backward_no_optimizer"] = "passed"
    report["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
    report["training_run"] = False
    (out / "smoke_results.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
