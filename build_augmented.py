"""Freeze reviewed public supplementation and exclude all original holdout rows."""
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageOps
from review_public import phash, dhash

ROOT = Path(__file__).resolve().parent
PUBLIC = ROOT/"data_external"


def write_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def main():
    reference = json.loads((ROOT/'runs/second/baseline_reference.json').read_text())
    with (ROOT/'data/labels.csv').open(encoding='utf8', newline='') as f:
        local = list(csv.DictReader(f))
    original_test = set(reference['validation_crops'])
    reserved_vehicles = {r['vehicle_id'] for r in local if r['crop'] in original_test}
    reserved_photos = {r['photo'] for r in local if r['crop'] in original_test}
    reviewed = json.loads((PUBLIC/'visual_review.json').read_text(encoding='utf8'))
    candidates = json.loads((PUBLIC/'review_manifest.json').read_text(encoding='utf8'))
    missing = [r['id'] for r in candidates if r['id'] not in reviewed]
    if missing:
        raise ValueError(f"{len(missing)} candidates still lack an explicit visual review: {missing[:8]}")
    # Compare against every local full photograph and active crop, not only holdout.
    local_hashes = []
    for path in list((ROOT/'cars_photos').glob('*.jpg')) + [ROOT/'data'/r['crop'] for r in local]:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert('RGB')
            local_hashes.append((str(path.relative_to(ROOT)), phash(im), dhash(im), hashlib.sha256(path.read_bytes()).hexdigest()))
    accepted, exclusions = [], []
    for r in candidates:
        decision = reviewed[r['id']]
        if not decision['accepted']:
            exclusions.append(dict(id=r['id'], reason=decision['reason'], stage='visual_review'))
            continue
        assert r['brand'] == decision['brand']
        assert 'creativecommons.org/' in r['license_url'] and r['author']
        p, d, full = int(r['crop_phash'], 16), int(r['crop_dhash'], 16), int(r['image_phash'], 16)
        matches = [name for name, lp, ld, sha in local_hashes
                   if sha == r['download_sha256'] or (p^lp).bit_count() <= 6 or (full^lp).bit_count() <= 6
                   or ((p^lp).bit_count() <= 12 and (d^ld).bit_count() <= 3)]
        if matches:
            exclusions.append(dict(id=r['id'], reason='Conservative local duplicate/near-duplicate match', matched=matches, stage='dedup_local'))
            continue
        duplicates = [a['id'] for a in accepted if a['download_sha256'] == r['download_sha256']
            or a['crop_sha256'] == r['crop_sha256'] or (p^int(a['crop_phash'],16)).bit_count() <= 6
            or (full^int(a['image_phash'],16)).bit_count() <= 6
            or ((p^int(a['crop_phash'],16)).bit_count() <= 12 and (d^int(a['crop_dhash'],16)).bit_count() <= 3)]
        if duplicates:
            exclusions.append(dict(id=r['id'], reason='Public duplicate/near-duplicate', matched=duplicates, stage='dedup_public'))
            continue
        # Group an author's entire brand collection conservatively for public dev.
        # This groups different models too, instead of pretending file IDs are vehicle IDs.
        group_key = r['brand']+'|'+r['author'].lower().strip()
        group = 'commons_author_brand_'+hashlib.sha256(group_key.encode()).hexdigest()[:16]
        accepted.append(dict(r, colour=decision['colour'], review_reason=decision['reason'],
            external_group=group, nearest_local_phash_distance=min(min((p^lp).bit_count(),(full^lp).bit_count()) for _,lp,_,_ in local_hashes),
            changes=r['changes']+'; EXIF-normalized RGB dominant car crop '+json.dumps(r['bbox'])+'; JPEG quality 95; no recolouring',
            label_evidence='Source title/description plus explicit Codex visual consistency review; no brand classifier'))
    # Public-only development split. Original Tashkent validation cannot select new epochs.
    by_group = defaultdict(list)
    for r in accepted:
        by_group[r['external_group']].append(r)
    dev_groups = set()
    rng = random.Random(3210)
    for brand in sorted({r['brand'] for r in accepted}):
        groups = sorted({r['external_group'] for r in accepted if r['brand'] == brand})
        total = sum(len(by_group[g]) for g in groups)
        rng.shuffle(groups)
        eligible = [g for g in groups if total-len(by_group[g]) >= 8 and len(groups) >= 2]
        if eligible:
            dev_groups.add(min(eligible, key=lambda g: abs(len(by_group[g])-.15*total)))
    manifest = []
    for r in local:
        role = 'benchmark' if r['crop'] in original_test else 'train'
        if role == 'train':
            assert r['vehicle_id'] not in reserved_vehicles and r['photo'] not in reserved_photos
        manifest.append(dict(id='tashkent_'+Path(r['crop']).stem, crop='data/'+r['crop'], brand=r['brand'], colour=r['colour'],
            domain='tashkent', role=role, group=r['source'], vehicle_id=r['vehicle_id'], photo=r['photo'],
            source='User-supplied Tashkent collection', evidence=r['evidence'],
            sha256=hashlib.sha256((ROOT/'data'/r['crop']).read_bytes()).hexdigest()))
    for r in accepted:
        manifest.append(dict(id=r['id'], crop='data_external/'+r['crop'], brand=r['brand'], colour=r['colour'],
            domain='commons', role='development' if r['external_group'] in dev_groups else 'train',
            group=r['external_group'], vehicle_id=None, photo='data_external/'+r['local_path'],
            source=r['source_page_url'], evidence=r['label_evidence'], sha256=r['crop_sha256']))
    assert len([r for r in manifest if r['role']=='benchmark']) == 10
    assert {r['group'] for r in manifest if r['role']=='train'}.isdisjoint({r['group'] for r in manifest if r['role'] in ('benchmark','development')})
    (PUBLIC/'accepted_manifest.json').write_text(json.dumps(accepted, indent=2, ensure_ascii=False), encoding='utf8')
    (PUBLIC/'exclusions.json').write_text(json.dumps(exclusions, indent=2, ensure_ascii=False), encoding='utf8')
    write_csv(PUBLIC/'ATTRIBUTIONS.csv', accepted, ['id','brand','colour','title','author','source_page_url','original_url','download_url','license','license_url','credit','changes','crop','crop_sha256','label_evidence'])
    write_csv(PUBLIC/'source_counts.csv', [dict(source='Wikimedia Commons',brand=b,images=sum(r['brand']==b for r in accepted),
        colour_labelled=sum(r['brand']==b and r['colour'] is not None for r in accepted),
        train=sum(r['brand']==b and r['external_group'] not in dev_groups for r in accepted),
        development=sum(r['brand']==b and r['external_group'] in dev_groups for r in accepted)) for b in sorted({r['brand'] for r in accepted})],
        ['source','brand','images','colour_labelled','train','development'])
    write_csv(PUBLIC/'license_counts.csv', [dict(license=l,images=sum(r['license']==l for r in accepted),url=next(r['license_url'] for r in accepted if r['license']==l)) for l in sorted({r['license'] for r in accepted})], ['license','images','url'])
    out = ROOT/'data_augmented'
    out.mkdir(exist_ok=True)
    payload = json.dumps(manifest, sort_keys=True)
    digest = hashlib.sha256(payload.encode()).hexdigest()
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf8')
    summary = dict(public_candidates=len(candidates), public_accepted=len(accepted),
        public_colour_labelled=sum(r['colour'] is not None for r in accepted),
        public_excluded=len(exclusions), public_duplicates_removed=sum(r['stage'].startswith('dedup') for r in exclusions),
        role_counts=dict(Counter(r['role'] for r in manifest)), source_counts=dict(Counter(r['source'] for r in accepted)),
        brands=dict(Counter(r['brand'] for r in accepted)), licences=dict(Counter(r['license'] for r in accepted)),
        original_holdout_crops=reference['validation_crops'],
        baseline_sha256=reference['baseline_sha256'], manifest_sha256=digest,
        identity_limitation='Commons has no globally reliable physical-vehicle IDs. Public dev uses conservative author+brand groups. Tashkent vehicle IDs remain frozen; all public images visually reviewed for foreign context and local SHA/pHash/dHash matches excluded.')
    (out/'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf8')
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == '__main__':
    main()
