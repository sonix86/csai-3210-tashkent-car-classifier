"""Paired final comparison on the unchanged original Tashkent holdout."""
import csv, hashlib, json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from finetune_augmented import Images
from project import COLOURS,load_model,CarDataset
from experiment import head_metrics,fingerprint

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'runs/second'

def main():
    ref=json.loads((OUT/'baseline_reference.json').read_text())
    assert hashlib.sha256((ROOT/ref['baseline_path']).read_bytes()).hexdigest()==ref['baseline_sha256']
    with (ROOT/'data/labels.csv').open(encoding='utf8') as f:local=list(csv.DictReader(f))
    assert fingerprint(local,ROOT/'data')==ref['original_dataset_fingerprint']
    with (ROOT/'.setup/original_hashes.csv').open(encoding='utf-8-sig') as f:
        originals=list(csv.DictReader(f))
    for r in originals:assert hashlib.sha256(Path(r['Path']).read_bytes()).hexdigest().upper()==r['Hash']
    rows=json.loads((ROOT/'data_augmented/manifest.json').read_text(encoding='utf8'))
    holdout=[r for r in rows if r['role']=='benchmark']
    assert {r['crop'][5:] for r in holdout}==set(ref['validation_crops'])
    models={};allbrands=sorted({r['brand'] for r in rows})
    for name,path in [('old',ROOT/'runs/first/best.pt'),('new',OUT/'best.pt')]:
        model,brands=load_model(path,'cuda');loader=DataLoader(CarDataset(holdout,ROOT,brands),batch_size=16)
        with torch.inference_mode():
            x,b,c=next(iter(loader));pb,pc=model(x.cuda());pb=pb.softmax(1).cpu().numpy();pc=pc.softmax(1).cpu().numpy()
        pred=[dict(id=r['id'],vehicle_id=r['vehicle_id'],photo=r['photo'],brand_true=r['brand'],brand_pred=brands[int(bp.argmax())],colour_true=r['colour'],colour_pred=COLOURS[int(cp.argmax())],brand_probabilities={b:float(p) for b,p in zip(brands,bp)},colour_probabilities=cp.tolist()) for r,bp,cp in zip(holdout,pb,pc)]
        bt=[allbrands.index(r['brand_true']) for r in pred];bp=[allbrands.index(r['brand_pred']) for r in pred];ct=[COLOURS.index(r['colour_true']) for r in pred];cp=[COLOURS.index(r['colour_pred']) for r in pred]
        vehicles=defaultdict(list)
        for r in pred:vehicles[r['vehicle_id']].append(r)
        vp=[]
        for v,rr in vehicles.items():
            bmean=np.mean([[r['brand_probabilities'].get(b,0) for b in allbrands] for r in rr],axis=0);cmean=np.mean([r['colour_probabilities'] for r in rr],axis=0)
            assert len({r['brand_true'] for r in rr})==1
            vp.append(dict(vehicle_id=v,brand_true=rr[0]['brand_true'],brand_pred=allbrands[int(bmean.argmax())],colour_true=rr[0]['colour_true'],colour_pred=COLOURS[int(cmean.argmax())]))
        metrics=dict(brand=head_metrics(bt,bp,allbrands),colour=head_metrics(ct,cp,COLOURS),joint_accuracy=sum(r['brand_true']==r['brand_pred'] and r['colour_true']==r['colour_pred'] for r in pred)/len(pred),vehicle_brand_accuracy=sum(r['brand_true']==r['brand_pred'] for r in vp)/len(vp),vehicle_colour_accuracy=sum(r['colour_true']==r['colour_pred'] for r in vp)/len(vp),vehicles=len(vp))
        models[name]=metrics
        (OUT/f'{name}_benchmark_predictions.json').write_text(json.dumps(pred,indent=2),encoding='utf8')
        (OUT/f'{name}_vehicle_predictions.json').write_text(json.dumps(vp,indent=2),encoding='utf8')
        for head,truth,predicted,classes in [('brand',bt,bp,allbrands),('colour',ct,cp,COLOURS)]:
            matrix=confusion_matrix(truth,predicted,labels=list(range(len(classes))))
            with (OUT/f'{name}_{head}_confusion.csv').open('w',newline='',encoding='utf-8-sig') as f:
                w=csv.writer(f);w.writerow(['true / predicted']+classes);w.writerows([[c]+row.tolist() for c,row in zip(classes,matrix)])
    result=dict(models=models,originals_unchanged=len(originals),holdout_crops=len(holdout),note='Same reused validation as first experiment; unseen by gradient training, but used to select first model. Not a fresh blind test. New checkpoint selected only on public development set.')
    (OUT/'comparison.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps({k:{'brand':v['brand']['accuracy'],'colour':v['colour']['accuracy'],'vehicle_brand':v['vehicle_brand_accuracy']} for k,v in models.items()},indent=2))

if __name__=='__main__':main()
