"""Prepare local release assets only. Never contacts or publishes to GitHub."""
import csv,hashlib,json,shutil,zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'.publication'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def archive(name,paths):
    target=OUT/name
    if not target.exists():
        with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for p in sorted(set(paths)):z.write(p,p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        assert set(z.namelist())=={p.relative_to(ROOT).as_posix() for p in paths}
        for p in paths:
            assert hashlib.sha256(z.read(p.relative_to(ROOT).as_posix())).hexdigest()==sha(p)
    return target

def main():
    OUT.mkdir(exist_ok=True)
    accepted=json.loads((ROOT/'data_external/accepted_manifest.json').read_text(encoding='utf8'))
    with (ROOT/'data/labels.csv').open(encoding='utf8') as f:local=list(csv.DictReader(f))
    public=[]
    for r in accepted:
        p=ROOT/'data_external'/r['crop'];assert sha(p)==r['crop_sha256']
        public.extend([p,p.with_suffix('.license.json')])
    public += [ROOT/'data_external'/n for n in ('ATTRIBUTIONS.csv','accepted_manifest.json','SOURCES_AND_LICENSES.md','source_counts.csv','license_counts.csv','brand_colour_counts.csv')]
    local_paths=list((ROOT/'cars_photos').glob('*.jpg'))+[ROOT/'data'/r['crop'] for r in local]
    local_paths += [ROOT/'data'/n for n in ('labels.csv','original_sha256.json','counts_crops.csv','counts_vehicles.csv','review_summary.json','split_preview.json')]
    assets=[archive('csai3210-commons-reviewed.zip',public),archive('csai3210-tashkent-originals-and-crops.zip',local_paths)]
    for run in ('first','second'):
        source=ROOT/f'runs/{run}/best.pt';target=OUT/f'csai3210-{run}-best.pt'
        if not target.exists():shutil.copy2(source,target)
        assert sha(source)==sha(target);assets.append(target)
    records=[dict(file=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in assets]
    (OUT/'SHA256SUMS.txt').write_text(''.join(f"{r['sha256']}  {r['file']}\n" for r in records),encoding='utf8')
    manifest=dict(status='Prepared locally; NOT uploaded. No release URL exists yet.',assets=records,
        restore={'csai3210-first-best.pt':['runs/first/best.pt','runs/second/baseline_best.pt'],'csai3210-second-best.pt':['runs/second/best.pt'],'zip_files':'Extract at repository root, preserving paths; never overwrite differing originals.'},
        dataset_permissions='Commons files keep individual licences and sidecars. Tashkent originals require owner approval before any upload. No blanket dataset licence is granted here.')
    existing=ROOT/'release_assets.json'
    if existing.exists():
        previous=json.loads(existing.read_text(encoding='utf8'))
        if previous.get('published'):
            assert previous['assets']==[dict(r,download_url=next(a['download_url'] for a in previous['assets'] if a['file']==r['file'])) for r in records]
            manifest=previous  # Never replace verified published URLs with staging status.
    existing.write_text(json.dumps(manifest,indent=2),encoding='utf8')
    print(json.dumps(records,indent=2))

if __name__=='__main__':main()
