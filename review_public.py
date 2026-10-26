"""Prepare licensed car candidates for explicit visual review; no brand model."""
import hashlib
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT/"data_external"
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT/".cache/ultralytics"))
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from ultralytics import YOLO


def phash(image):
    gray = np.array(ImageOps.grayscale(image).resize((32, 32)), dtype=np.float32)
    coefficients = cv2.dct(gray)[:8, :8].flatten()
    bits = coefficients > np.median(coefficients[1:])
    return int(''.join('1' if b else '0' for b in bits), 2)


def dhash(image):
    a = np.array(ImageOps.grayscale(image).resize((9, 8)))
    return int(''.join('1' if b else '0' for b in (a[:, 1:] > a[:, :-1]).flatten()), 2)


def main():
    records = []
    for path in (OUT/"metadata").glob('*.json'):
        records += json.loads(path.read_text(encoding="utf8"))
    records.sort(key=lambda r: (r["brand"], r["id"]))
    cache = OUT/"detection_review.json"
    done = json.loads(cache.read_text(encoding="utf8")) if cache.exists() else {}
    detector = YOLO(str(ROOT/"yolo11n.pt"))
    (OUT/"crops").mkdir(exist_ok=True)
    for number, r in enumerate(records):
        if r["id"] in done or not (OUT/r["local_path"]).is_file():
            continue
        try:
            aliases = {"Deepal": ["Shenlan", "Deepal"], "Skoda": ["Skoda", "Škoda"]}.get(r["brand"], [r["brand"]])
            if not any(re.search(r'\b'+re.escape(a)+r'\b', r["title"], re.I) for a in aliases):
                done[r["id"]] = dict(r, status="rejected", reason="No exact brand token in file title")
                continue
            with Image.open(OUT/r["local_path"]) as im:
                image = ImageOps.exif_transpose(im).convert("RGB")
            result = detector.predict(source=np.array(image)[:, :, ::-1].copy(), classes=[2], conf=.5, device=0, verbose=False)[0]
            boxes = []
            for box in result.boxes:
                coords = list(map(int, box.xyxy[0].tolist()))
                area = max(0, coords[2]-coords[0])*max(0, coords[3]-coords[1])
                boxes.append((area, coords, float(box.conf.item())))
            boxes.sort(reverse=True)
            if not boxes or boxes[0][0] < .15*image.width*image.height or min(boxes[0][1][2]-boxes[0][1][0],boxes[0][1][3]-boxes[0][1][1]) < 100:
                done[r["id"]] = dict(r, status="rejected", reason="No sufficiently large clear COCO car")
                continue
            if len(boxes) > 1 and boxes[1][0] > .65*boxes[0][0]:
                done[r["id"]] = dict(r, status="rejected", reason="Multiple similarly large cars; source label ambiguous")
                continue
            _, box, confidence = boxes[0]
            x1, y1, x2, y2 = box
            pad = round(.02*max(x2-x1, y2-y1))
            box = [max(0, x1-pad), max(0, y1-pad), min(image.width, x2+pad), min(image.height, y2+pad)]
            crop = image.crop(box)
            path = OUT/"crops"/(r["id"]+".jpg")
            crop.save(path, quality=95)
            done[r["id"]] = dict(r, status="pending_visual_review", crop="crops/"+path.name,
                bbox=box, detection_confidence=confidence,
                image_phash=f"{phash(image):016x}", crop_phash=f"{phash(crop):016x}", crop_dhash=f"{dhash(crop):016x}",
                download_sha256=hashlib.sha256((OUT/r["local_path"]).read_bytes()).hexdigest(),
                crop_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            if len(done) % 20 == 0:
                cache.write_text(json.dumps(done, indent=2, ensure_ascii=False), encoding="utf8")
                print("Prepared", len(done), "of", len(records), flush=True)
        except Exception as exc:
            print("Pending", r["id"], type(exc).__name__, flush=True)
    cache.write_text(json.dumps(done, indent=2, ensure_ascii=False), encoding="utf8")
    # Review indices are stable within each brand once all downloads are present.
    manifest = []
    for brand in sorted({r["brand"] for r in records}):
        candidates = sorted([r for r in done.values() if r["brand"] == brand and r["status"] == "pending_visual_review"], key=lambda r:r["id"])
        for i, record in enumerate(candidates, 1):
            manifest.append(dict(record, review_index=i))
        for start in range(0, len(candidates), 16):
            sheet = Image.new("RGB", (1400, 1160), "white")
            draw = ImageDraw.Draw(sheet)
            for j, record in enumerate(candidates[start:start+16]):
                x, y = j%4*350, j//4*290
                with Image.open(OUT/record["crop"]) as crop:
                    thumb = ImageOps.contain(crop, (345, 240))
                    sheet.paste(thumb, (x, y))
                draw.text((x+3, y+242), f"{start+j+1}: {brand} {record['id']}", fill="black")
                ascii_title = record["title"].encode('ascii','replace').decode()
                draw.text((x+3, y+258), ascii_title[:49], fill="black")
            sheet.save(OUT/"review"/(brand.replace(' ', '_')+f"_{start//16+1}.jpg"))
    (OUT/"review_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf8")
    print("Ready for visual review:",len(manifest), flush=True)


if __name__ == "__main__":
    main()
