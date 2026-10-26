"""List exactly what Git would add using .gitignore, without initializing project Git."""
import csv,json,subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parent
def main():
    temporary=ROOT/'.setup/publication-audit.git'
    if not temporary.exists():subprocess.run(['git','init','--bare',str(temporary)],check=True,capture_output=True)
    command=['git','--git-dir='+str(temporary),'--work-tree='+str(ROOT),'ls-files','--others','--exclude-standard','-z']
    result=subprocess.run(command,cwd=ROOT,check=True,capture_output=True)
    paths=sorted(p for p in result.stdout.decode('utf8').split('\0') if p)
    for name in ('PUBLICATION_FILES.csv','PUBLICATION_CHECKS.json'):
        if name not in paths:paths.append(name)
    records=[dict(path=p,bytes=(ROOT/p).stat().st_size if (ROOT/p).exists() else 0) for p in sorted(paths)]
    forbidden=('cars_photos/','data/crops/','data_external/crops/','data_external/downloads/','.venv/','.python/','.cache/','.setup/','.publication/')
    assert not any(r['path'].startswith(forbidden) or r['path'].endswith('.pt') for r in records)
    for required in ('runs/first/REPORT.md','runs/first/metrics.json','runs/first/history.csv','runs/second/REPORT.md','runs/second/comparison.json','runs/second/history.csv','data_external/ATTRIBUTIONS.csv'):
        assert required in paths,required
    assert not any(r['bytes']>=100*1024**2 for r in records)
    with (ROOT/'PUBLICATION_FILES.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['path','bytes']);w.writeheader();w.writerows(records)
    checks=dict(files=len(records),bytes=sum(r['bytes'] for r in records),largest=sorted(records,key=lambda r:r['bytes'],reverse=True)[:10],no_models_or_photos_in_git=True,historical_metrics_included=True,github_cli='Not installed / not on PATH; gh auth status could not execute',published=False)
    (ROOT/'PUBLICATION_CHECKS.json').write_text(json.dumps(checks,indent=2),encoding='utf8')
    print(json.dumps(checks,indent=2))

if __name__=='__main__':main()
