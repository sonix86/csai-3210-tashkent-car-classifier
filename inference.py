"""Local COCO car detection and the project's own brand/colour heads."""
import json
import time
from pathlib import Path

import cv2
import torch
from PIL import Image


def synchronize(device):
    if device == "cuda":
        torch.cuda.synchronize()


def protect_original(path, input_path=None):
    path = Path(path).resolve()
    originals = Path(__file__).resolve().parent/"cars_photos"
    if path.is_relative_to(originals) or (input_path and path == Path(input_path).resolve()):
        raise ValueError("Refusing to overwrite an original/input photograph")


class CarRecognizer:
    def __init__(self, weights, detector, device="auto"):
        from experiment import choose_device
        from project import load_model, image_transform
        from ultralytics import YOLO
        if not Path(detector).is_file():
            raise FileNotFoundError("Local COCO detector not found; cache weights before offline inference")
        self.device = choose_device(device)
        self.model, self.brands = load_model(weights, self.device)
        self.detector = YOLO(str(detector))
        self.transform = image_transform(False)

    def infer(self, frame, detect_conf=.25, min_size=35, brand_threshold=.4):
        from project import COLOURS
        synchronize(self.device)
        started = time.perf_counter()
        result = self.detector.predict(source=frame, classes=[2], conf=detect_conf, verbose=False,
            device=0 if self.device == "cuda" else "cpu")[0]
        height, width = frame.shape[:2]
        candidates = []
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
            if x2-x1 >= min_size and y2-y1 >= min_size:
                candidates.append(([x1, y1, x2, y2], float(box.conf.item())))
        candidates.sort(key=lambda item: (item[0][1], item[0][0]))
        synchronize(self.device)
        detect_seconds = time.perf_counter()-started
        cars = []
        if candidates:
            # Read every crop from the untouched input; annotations are drawn separately.
            batch = torch.stack([self.transform(Image.fromarray(cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)))
                for (x1, y1, x2, y2), confidence in candidates]).to(self.device)
            with torch.inference_mode():
                bp, cp = self.model(batch)
                bp, cp = bp.softmax(1).cpu(), cp.softmax(1).cpu()
            for index, ((box, confidence), b, c) in enumerate(zip(candidates, bp, cp), 1):
                bi, ci = int(b.argmax()), int(c.argmax())
                cars.append({"index": index, "box": box, "detection_confidence": confidence,
                    "brand": self.brands[bi] if float(b[bi]) >= brand_threshold else "unknown",
                    "brand_top1": self.brands[bi], "colour": COLOURS[ci],
                    "brand_confidence": float(b[bi]), "colour_confidence": float(c[ci])})
        synchronize(self.device)
        total = time.perf_counter()-started
        return {"cars": cars, "device": self.device, "detector_seconds": detect_seconds,
                "classifier_and_preprocessing_seconds": total-detect_seconds, "inference_seconds": total,
                "brand_threshold": brand_threshold}


def save_annotated(frame, cars, path, input_path=None):
    protect_original(path, input_path)
    annotated = frame.copy()
    for car in cars:
        x1, y1, x2, y2 = car["box"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (40, 230, 80), 2)
        cv2.putText(annotated, f"{car['index']}: {car['brand']} {car['colour']}", (x1, max(18, y1-7)),
                    cv2.FONT_HERSHEY_SIMPLEX, .6, (40, 230, 80), 2)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)).save(path)


def predict(args):
    from project import read_image
    started = time.perf_counter()
    if args.save:
        protect_original(args.save, args.image)
    json_path = getattr(args, "json", None)
    if json_path:
        protect_original(json_path, args.image)
    recognizer = CarRecognizer(args.weights, args.detector, getattr(args, "device", "auto"))
    frame = read_image(args.image)
    report = recognizer.infer(frame, args.detect_conf, args.min_size, args.brand_threshold)
    report["image"] = str(args.image)
    for car in report["cars"]:
        print(f"Car {car['index']}: {car['brand']}; {car['colour']} | top1={car['brand_top1']} brand_conf={car['brand_confidence']:.3f} colour_conf={car['colour_confidence']:.3f}")
    if not report["cars"]:
        print("No classifiable cars detected")
    if args.save:
        save_annotated(frame, report["cars"], args.save, args.image)
        print("Annotated image:", args.save)
    report["handler_wall_seconds_including_load_read_save"] = time.perf_counter()-started
    if json_path:
        Path(json_path).parent.mkdir(parents=True, exist_ok=True)
        Path(json_path).write_text(json.dumps(report, indent=2), encoding="utf8")
    print(f"Inference: {report['inference_seconds']:.3f}s; handler with load/read/save: {report['handler_wall_seconds_including_load_read_save']:.3f}s")
    return report
