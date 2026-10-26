"""Add model diversity missed by broad Commons relevance search."""
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from collect_public import OUT, discover, download

QUERIES = [
    ('Deepal', 'intitle:Shenlan'), ('Skoda', 'intitle:Skoda intitle:Octavia'),
    ('BYD', 'intitle:BYD intitle:Han'), ('Chevrolet', 'intitle:Chevrolet intitle:Spark'),
    ('BMW', 'intitle:BMW intitle:X7'), ('Kia', 'intitle:Kia intitle:Sorento'),
    ('Tesla', 'intitle:Tesla intitle:Model'), ('Toyota', 'intitle:Toyota intitle:Camry')]


def main():
    rows = []
    for brand, query in QUERIES:
        key = brand.replace(' ', '_')+'_diversity'
        records = discover(key, query, 20)
        for r in records:
            r['brand'] = brand
        (OUT/'metadata'/(key+'.json')).write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf8')
        rows += records
    errors, complete = [], []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(download, r): r for r in rows}
        for future in as_completed(futures):
            try:
                complete.append(future.result())
            except Exception as exc:
                errors.append(dict(id=futures[future]['id'], error=str(exc)))
    (OUT/'additional_candidates.json').write_text(json.dumps(complete,ensure_ascii=False,indent=2),encoding='utf8')
    (OUT/'additional_download_errors.json').write_text(json.dumps(errors,ensure_ascii=False,indent=2),encoding='utf8')
    print('Additional downloads',len(complete),'errors',len(errors),flush=True)


if __name__ == '__main__':
    main()
