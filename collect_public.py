"""Collect Commons candidates with explicit file-level reuse metadata; no models."""
import argparse
import hashlib
import html
import json
import re
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parent
OUT = ROOT/"data_external"
API = "https://commons.wikimedia.org/w/api.php"
HEADERS = {"User-Agent": "CSAI3210-AcademicDataset/1.0 (educational image reuse audit; file attribution retained)"}
QUERIES = {
    "Aion": 'intitle:Aion', "BMW": 'intitle:BMW (X5 OR X7 OR X3 OR sedan)',
    "BYD": 'intitle:BYD', "Chery": 'intitle:Chery',
    "Chevrolet": 'intitle:Chevrolet (Spark OR Cobalt OR Nexia OR Lacetti OR Tracker OR Trailblazer)',
    "Deepal": 'intitle:Shenlan', "Haval": 'intitle:Haval',
    "Kia": 'intitle:Kia', "Leapmotor": 'intitle:Leapmotor', "Li Auto": 'intitle:"Li Auto"',
    "Skoda": 'intitle:Skoda (Rapid OR Octavia OR Superb OR Kodiaq OR Karoq)', "Tesla": 'intitle:Tesla',
    "Toyota": 'intitle:Toyota (Camry OR Corolla OR RAV4)', "Voyah": 'intitle:Voyah',
    "Zeekr": 'intitle:Zeekr', "Changan": 'intitle:Changan', "Geely": 'intitle:Geely',
    "Jetour": 'intitle:Jetour'}
EXCLUDE = re.compile(r'\b(interior|engine|logo|badge|emblem|wheel|dashboard|steering|cockpit|seat|taillight|headlight|bus|truck|diagram|drawing|scale model|toy|Uzbekistan|Tashkent)\b', re.I)


def plain(value):
    return html.unescape(re.sub('<[^>]*>', '', value or '')).strip()


def get_json(params):
    for attempt in range(5):
        response = requests.get(API, params={"format": "json", **params}, headers=HEADERS, timeout=45)
        if response.status_code in (429, 503):
            time.sleep(3*(attempt+1))
            continue
        response.raise_for_status()
        result = response.json()
        if "error" in result:
            raise RuntimeError(result["error"])
        return result
    raise RuntimeError("Commons request rate limited repeatedly")


def discover(brand, query, limit):
    cache = OUT/"metadata"/(brand.replace(' ', '_')+".json")
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf8"))
    result = get_json({"action": "query", "generator": "search", "gsrnamespace": 6,
        "gsrlimit": min(limit, 50), "gsrsearch": query+' filetype:bitmap -intitle:interior -intitle:logo -intitle:engine',
        "prop": "imageinfo", "iiprop": "url|extmetadata|size|sha1", "iiurlwidth": 960})
    candidates = []
    for page in result.get("query", {}).get("pages", {}).values():
        title = page["title"]
        info = page.get("imageinfo", [{}])[0]
        meta = info.get("extmetadata", {})
        license_name = plain(meta.get("LicenseShortName", {}).get("value", ""))
        license_url = plain(meta.get("LicenseUrl", {}).get("value", ""))
        author = plain(meta.get("Artist", {}).get("value", ""))
        if not (re.fullmatch(r'CC BY(?:-SA)? [1-4]\.0', license_name) or license_name == "CC0"):
            continue
        if "creativecommons.org/" not in license_url or not author:
            continue
        if EXCLUDE.search(title) or min(info.get("width", 0), info.get("height", 0)) < 400:
            continue
        if not re.search(r'\.(jpe?g|png)$', title, re.I):
            continue
        identifier = f"commons_{page['pageid']}"
        candidates.append({"id": identifier, "brand": brand, "title": title,
            "pageid": page["pageid"], "source": "Wikimedia Commons", "source_page_url": info["descriptionurl"],
            "original_url": info["url"], "download_url": info.get("thumburl", info["url"]),
            "license": license_name, "license_url": license_url, "author": author,
            "artist_html": meta.get("Artist", {}).get("value", ""),
            "credit": plain(meta.get("Credit", {}).get("value", "")),
            "copyrighted": plain(meta.get("Copyrighted", {}).get("value", "")),
            "description": plain(meta.get("ImageDescription", {}).get("value", ""))[:1200],
            "commons_original_sha1": info.get("sha1"), "original_width": info["width"], "original_height": info["height"],
            "local_path": "downloads/"+identifier+".jpg", "retrieved_at": "2026-10-26",
            "changes": "Commons server thumbnail resized to max width 960; later car crop recorded separately",
            "label_evidence": "File title and source description; pending visual consistency review"})
    cache.write_text(json.dumps(candidates, indent=2, ensure_ascii=False), encoding="utf8")
    print(brand, "licensed candidates:", len(candidates), flush=True)
    return candidates


def download(record):
    path = OUT/record["local_path"]
    if not path.exists():
        for attempt in range(3):
            r = requests.get(record["download_url"], headers=HEADERS, timeout=60)
            if r.status_code == 429:
                time.sleep(5*(attempt+1))
                continue
            r.raise_for_status()
            path.write_bytes(r.content)
            break
    with Image.open(path) as im:
        im.verify()
    record["download_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--brands", nargs="+")
    args = parser.parse_args()
    for name in ("metadata", "downloads", "review"):
        (OUT/name).mkdir(parents=True, exist_ok=True)
    records = []
    errors = []
    for brand, query in QUERIES.items():
        if args.brands and brand not in args.brands:
            continue
        try:
            records.extend(discover(brand, query, args.limit))
        except Exception as exc:
            errors.append({"brand": brand, "error": str(exc)})
            print(brand, "discovery error", str(exc), flush=True)
        time.sleep(.25)
    # Two workers only; honor rate limiting and cache successful downloads.
    completed = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(download, r): r for r in records}
        for future in as_completed(futures):
            r = futures[future]
            try:
                completed.append(future.result())
            except Exception as exc:
                errors.append({"id": r["id"], "brand": r["brand"], "error": str(exc)})
            if (len(completed)+len(errors)) % 30 == 0:
                print("Downloaded", len(completed), "/", len(records), "errors", len(errors), flush=True)
    if args.brands and (OUT/"candidates.json").exists():
        completed += [r for r in json.loads((OUT/"candidates.json").read_text(encoding="utf8")) if r["brand"] not in args.brands]
    (OUT/"candidates.json").write_text(json.dumps(sorted(completed, key=lambda r: (r["brand"], r["id"])), ensure_ascii=False, indent=2), encoding="utf8")
    (OUT/"download_errors.json").write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf8")
    print("Total downloaded:", len(completed), "errors:", len(errors), flush=True)


if __name__ == "__main__":
    main()
