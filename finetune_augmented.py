"""One fixed CUDA warm-start experiment; benchmark is evaluated only afterwards."""
import csv, hashlib, json, random, time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image, ImageOps
from torchvision import transforms as T
from sklearn.metrics import balanced_accuracy_score
from project import DualHeadRegNet, COLOURS

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'runs/second'

def warm_start(checkpoint, brands):
    model=DualHeadRegNet(len(brands))
    state=model.state_dict()
    for k,v in checkpoint['model'].items():
        if k not in ('brand_head.3.weight','brand_head.3.bias'): state[k]=v
    for i,b in enumerate(checkpoint['brands']):
        j=brands.index(b)
        state['brand_head.3.weight'][j]=checkpoint['model']['brand_head.3.weight'][i]
        state['brand_head.3.bias'][j]=checkpoint['model']['brand_head.3.bias'][i]
    model.load_state_dict(state)
    return model

class Images(Dataset):
    def __init__(self,rows,brands,training=False):
        self.rows=rows;self.brands=brands;self.images=[]
        for r in rows:
            with Image.open(ROOT/r['crop']) as im:self.images.append(ImageOps.contain(ImageOps.exif_transpose(im).convert('RGB'),(480,480)))
        aug=[T.RandomResizedCrop(224,scale=(.8,1.),ratio=(.85,1.18)),T.RandomHorizontalFlip(),T.RandomRotation(5),T.ColorJitter(brightness=.15,contrast=.15)] if training else [T.Resize((224,224))]
        self.transform=T.Compose(aug+[T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i];return self.transform(self.images[i]),self.brands.index(r['brand']),COLOURS.index(r['colour']) if r['colour'] is not None else -100

def masked_loss(bp,cp,b,c,smoothing=.1):
    loss=.7*nn.functional.cross_entropy(bp,b,label_smoothing=smoothing)
    valid=c!=-100
    if valid.any():loss=loss+.3*nn.functional.cross_entropy(cp[valid],c[valid],label_smoothing=smoothing)
    return loss

@torch.inference_mode()
def assess(model,loader):
    model.eval();bt=[];ct=[];bp=[];cp=[];loss=0
    for x,b,c in loader:
        pb,pc=model(x.cuda());loss+=float(masked_loss(pb,pc,b.cuda(),c.cuda(),0))*len(b)
        bt+=b.tolist();ct+=c.tolist();bp+=pb.argmax(1).cpu().tolist();cp+=pc.argmax(1).cpu().tolist()
    valid=np.array(ct)!=-100
    return dict(loss=loss/len(bt),brand_balanced_accuracy=float(balanced_accuracy_score(bt,bp)),colour_balanced_accuracy=float(balanced_accuracy_score(np.array(ct)[valid],np.array(cp)[valid])))

def main():
    if (OUT/'best.pt').exists():raise FileExistsError('Preserve completed run')
    assert torch.cuda.is_available()
    torch.set_num_threads(8);torch.manual_seed(3210);random.seed(3210);np.random.seed(3210)
    rows=json.loads((ROOT/'data_augmented/manifest.json').read_text(encoding='utf8'))
    summary=json.loads((ROOT/'data_augmented/summary.json').read_text(encoding='utf8'))
    for r in rows:
        assert hashlib.sha256((ROOT/r['crop']).read_bytes()).hexdigest()==r['sha256']
        assert r['colour'] is None or r['colour'] in COLOURS
    train=[r for r in rows if r['role']=='train'];dev=[r for r in rows if r['role']=='development'];test=[r for r in rows if r['role']=='benchmark']
    assert {r['group'] for r in train}.isdisjoint({r['group'] for r in dev+test})
    brands=sorted({r['brand'] for r in train});old=torch.load(OUT/'baseline_best.pt',weights_only=True,map_location='cpu')
    assert hashlib.sha256((OUT/'baseline_best.pt').read_bytes()).hexdigest()==summary['baseline_sha256']
    config=dict(epochs=40,batch=16,seed=3210,body_lr=1e-5,head_lr=3e-4,freeze_body_epochs=8,freeze_batchnorm=True,batchnorm_recalibration='training images only, before optimization',label_smoothing=.1,weight_decay=.03,colour_missing='masked, never pseudo-labelled',sampling='inverse brand frequency, local domain multiplier 3, inverse sqrt group size',selection='0.7 public dev brand balanced accuracy + 0.3 public dev colour balanced accuracy; loss tie break',benchmark_used_for_selection=False,warm_start='own runs/first/best.pt; expanded brand head by name',pretrained_brand_classifier=False,gpu=torch.cuda.get_device_name(),torch=str(torch.__version__),manifest_sha256=summary['manifest_sha256'])
    (OUT/'config.json').write_text(json.dumps(config,indent=2),encoding='utf8')
    (OUT/'split.json').write_text(json.dumps({role:[r['id'] for r in rows if r['role']==role] for role in ('train','development','benchmark')},indent=2),encoding='utf8')
    counts=Counter(r['brand'] for r in train);groups=Counter(r['group'] for r in train)
    weights=[(3 if r['domain']=='tashkent' else 1)/counts[r['brand']]/groups[r['group']]**.5 for r in train]
    sampler=WeightedRandomSampler(weights,max(600,len(train)),replacement=True,generator=torch.Generator().manual_seed(3210))
    loader=DataLoader(Images(train,brands,True),batch_size=16,sampler=sampler,num_workers=0)
    devloader=DataLoader(Images(dev,brands),batch_size=32)
    model=warm_start(old,brands).cuda()
    # Re-estimate old narrow-domain BN statistics using only expanded train images.
    model.eval()
    for mod in model.modules():
        if isinstance(mod,nn.modules.batchnorm._BatchNorm):mod.reset_running_stats();mod.momentum=None;mod.train()
    with torch.no_grad():
        for x,_,_ in DataLoader(Images(train,brands),batch_size=32):model(x.cuda())
    optimizer=torch.optim.AdamW([dict(params=model.body.parameters(),lr=1e-5),dict(params=list(model.brand_head.parameters())+list(model.colour_head.parameters()),lr=3e-4)],weight_decay=.03)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,40);scaler=torch.amp.GradScaler('cuda')
    best=(-1.,-float('inf'));history=[];torch.cuda.synchronize();start=time.perf_counter()
    for epoch in range(1,41):
        tick=time.perf_counter();model.train()
        for p in model.body.parameters():p.requires_grad_(epoch>8)
        for mod in model.modules():
            if isinstance(mod,nn.modules.batchnorm._BatchNorm):mod.eval()
        total=0;n=0
        for x,b,c in loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda'):
                pb,pc=model(x.cuda());loss=masked_loss(pb,pc,b.cuda(),c.cuda())
            assert torch.isfinite(loss)
            scaler.scale(loss).backward();scaler.unscale_(optimizer);nn.utils.clip_grad_norm_(model.parameters(),5);scaler.step(optimizer);scaler.update()
            total+=float(loss.detach())*len(b);n+=len(b)
        metrics=assess(model,devloader);key=(.7*metrics['brand_balanced_accuracy']+.3*metrics['colour_balanced_accuracy'],-metrics['loss'])
        if key>best:
            best=key;torch.save(dict(model={k:v.cpu().clone() for k,v in model.state_dict().items()},brands=brands,colours=COLOURS,config=config,best_epoch=epoch,baseline_sha256=summary['baseline_sha256'],manifest_sha256=summary['manifest_sha256'],development_metrics=metrics),OUT/'best.pt')
        scheduler.step();record=dict(epoch=epoch,train_loss=total/n,**metrics,seconds=time.perf_counter()-tick);history.append(record)
        with (OUT/'history.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(record));w.writeheader();w.writerows(history)
        print(json.dumps(record),flush=True)
    torch.cuda.synchronize();timing=dict(training_seconds=time.perf_counter()-start,weights_bytes=(OUT/'best.pt').stat().st_size,peak_cuda_bytes=torch.cuda.max_memory_allocated())
    (OUT/'training_summary.json').write_text(json.dumps(timing,indent=2));print(timing,flush=True)

if __name__=='__main__':main()
