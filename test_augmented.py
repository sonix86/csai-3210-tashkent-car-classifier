import hashlib,json,unittest
from pathlib import Path
import torch
from finetune_augmented import warm_start,masked_loss
from project import COLOURS

ROOT=Path(__file__).resolve().parent
class AugmentedChecks(unittest.TestCase):
    def test_named_warm_start(self):
        old=torch.load(ROOT/'runs/second/baseline_best.pt',weights_only=True,map_location='cpu')
        brands=sorted(old['brands']+['Changan','Geely','Jetour']);model=warm_start(old,brands)
        for i,b in enumerate(old['brands']):self.assertTrue(torch.equal(model.brand_head[3].weight[brands.index(b)],old['model']['brand_head.3.weight'][i]))
    def test_missing_colour_loss(self):
        bp=torch.randn(3,18,requires_grad=True);cp=torch.randn(3,8,requires_grad=True)
        loss=masked_loss(bp,cp,torch.tensor([0,1,2]),torch.tensor([-100,-100,-100]));loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertIsNone(cp.grad)
    def test_manifest_and_licences(self):
        rows=json.loads((ROOT/'data_augmented/manifest.json').read_text(encoding='utf8'))
        train=[r for r in rows if r['role']=='train'];held=[r for r in rows if r['role']!='train']
        self.assertFalse({r['group'] for r in train}&{r['group'] for r in held})
        self.assertFalse({r['sha256'] for r in train}&{r['sha256'] for r in held})
        self.assertEqual(sum(r['role']=='benchmark' for r in rows),10)
        for r in rows:self.assertIn(r['colour'],COLOURS+[None]);self.assertEqual(hashlib.sha256((ROOT/r['crop']).read_bytes()).hexdigest(),r['sha256'])
        for r in json.loads((ROOT/'data_external/accepted_manifest.json').read_text(encoding='utf8')):
            self.assertTrue(r['author']);self.assertIn('creativecommons.org/',r['license_url']);self.assertIn('commons.wikimedia.org/',r['source_page_url'])

if __name__=='__main__':unittest.main()
