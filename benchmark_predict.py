"""Benchmark two fixed multi-car photos; no fitting or threshold tuning."""
import csv
import hashlib
import json
import statistics
import time
from pathlib import Path

from inference import CarRecognizer, save_annotated, synchronize
from project import read_image


def iou(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0]))*max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union else 0.


def main():
    root = Path(__file__).resolve().parent
    out = root/"runs/first"
    with (root/"data/labels.csv").open(encoding="utf8", newline="") as f:
        labels = list(csv.DictReader(f))
    split = json.loads((out/"split.json").read_text())
    val = set(split["val_crops"])
    tick = time.perf_counter()
    engine = CarRecognizer(out/"best.pt", root/"yolo11n.pt", "cuda")
    synchronize("cuda")
    load_seconds = time.perf_counter()-tick
    cases = []
    for number in (27, 76):
        relative = f"cars_photos/photo_{number}_2026-10-26_17-10-50.jpg"
        path = root/relative
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        read_started = time.perf_counter()
        frame = read_image(path)
        read_seconds = time.perf_counter()-read_started
        raw_hash = hashlib.sha256(frame.tobytes()).hexdigest()
        first = engine.infer(frame)
        engine.infer(frame)  # second warm-up; all timed repetitions use fixed weights/thresholds
        elapsed = [engine.infer(frame)["inference_seconds"] for _ in range(10)]
        assert hashlib.sha256(frame.tobytes()).hexdigest() == raw_hash, "Inference modified input array"
        annotated = out/f"predict_photo_{number}.jpg"
        tick = time.perf_counter()
        save_annotated(frame, first["cars"], annotated, path)
        save_seconds = time.perf_counter()-tick
        truth = [r for r in labels if r["photo"] == relative]
        available = set(range(len(first["cars"])))
        matches = []
        for row in truth:
            target = json.loads(row["bbox"])
            candidates = sorted(available, key=lambda i: iou(target, first["cars"][i]["box"]), reverse=True)
            index = candidates[0] if candidates else None
            overlap = iou(target, first["cars"][index]["box"]) if index is not None else 0.
            car = first["cars"][index] if overlap >= .5 else None
            if car:
                available.remove(index)
            matches.append({"vehicle_id": row["vehicle_id"], "ground_truth_brand": row["brand"],
                "ground_truth_colour": row["colour"], "matched_iou": overlap,
                "prediction": car, "brand_top1_correct": car["brand_top1"] == row["brand"] if car else None,
                "displayed_brand_correct": car["brand"] == row["brand"] if car else None,
                "colour_correct": car["colour"] == row["colour"] if car else None})
        assert hashlib.sha256(path.read_bytes()).hexdigest() == before
        cases.append({"image": relative, "annotation_subset": "validation" if all(r["crop"] in val for r in truth) else "training",
            "annotated_objects": len(truth), "detected_cars": len(first["cars"]), "matches_iou_0_5": matches,
            "cars": first["cars"], "read_seconds": read_seconds, "save_seconds": save_seconds,
            "first_inference_seconds": first["inference_seconds"], "warmup_calls": 2, "timed_repetitions": 10,
            "warm_inference_median_seconds": statistics.median(elapsed),
            "warm_inference_min_seconds": min(elapsed), "warm_inference_max_seconds": max(elapsed),
            "timed_seconds": elapsed, "annotated_output": str(annotated), "original_unchanged": True})
    report = {"device": "cuda", "model_load_seconds": load_seconds, "brand_threshold": .4,
        "weights_sha256": hashlib.sha256((out/"best.pt").read_bytes()).hexdigest(), "cases": cases,
        "timing_scope": "GPU synchronized. Warm inference includes detector, transforms and batched classifier; excludes model loading, image disk read/write and process startup.",
        "limitation": "Photo 27 is a training demonstration, photo 76 is held-out validation. Only annotated objects are matched; background annotations are incomplete. No detection mAP or generalization claim."}
    (out/"prediction_benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
