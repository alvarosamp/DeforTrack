"""Audit visual overlap across dataset splits with ResNet-18 cosine similarity."""
import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18, ResNet18_Weights

ROLES = {"train": "train", "validation": "test", "test": "valid"}

class Images(Dataset):
    def __init__(self, root):
        self.paths=[]
        for role, folder in ROLES.items():
            self.paths += [(role,p) for p in sorted((root/folder/"images").glob("*")) if p.suffix.lower() in {".jpg",".jpeg",".png"}]
        self.transform=ResNet18_Weights.DEFAULT.transforms()
    def __len__(self): return len(self.paths)
    def __getitem__(self,i):
        role,p=self.paths[i]
        with Image.open(p) as im: x=self.transform(im.convert("RGB"))
        return x,role,str(p)

def extract(root,batch,workers,device):
    ds=Images(root); dl=DataLoader(ds,batch_size=batch,num_workers=workers,shuffle=False)
    model=resnet18(weights=ResNet18_Weights.DEFAULT); model.fc=torch.nn.Identity(); model.eval().to(device)
    feats=[]; roles=[]; paths=[]
    with torch.inference_mode():
        for i,(x,r,p) in enumerate(dl,1):
            z=torch.nn.functional.normalize(model(x.to(device)),dim=1).cpu()
            feats.append(z); roles.extend(r); paths.extend(p)
            if i%20==0: print(f"embedding batch {i}/{len(dl)}",flush=True)
    return torch.cat(feats),np.asarray(roles),np.asarray(paths)

def compare(a,b,fa,fb,pa,pb,chunk=256):
    thresholds=(0.90,0.95,0.98,0.99); counts={str(t):0 for t in thresholds}; nearest=[]; top=[]
    for start in range(0,len(fa),chunk):
        sim=fa[start:start+chunk]@fb.T
        vals,idx=sim.max(1)
        for j,(v,k) in enumerate(zip(vals.tolist(),idx.tolist())):
            nearest.append({"comparison":f"{a}-{b}","query_split":a,"query":pa[start+j],"reference_split":b,"reference":pb[k],"cosine":v})
        for t in thresholds: counts[str(t)]+=int((sim>=t).sum())
        flat=sim.flatten(); n=min(100,flat.numel()); v,k=torch.topk(flat,n)
        for score,pos in zip(v.tolist(),k.tolist()):
            i=pos//sim.shape[1]; j=pos%sim.shape[1]
            top.append((score,pa[start+i],pb[j]))
    top=sorted(top,reverse=True)[:100]
    scores=np.asarray([x["cosine"] for x in nearest])
    summary={"comparison":f"{a}-{b}","candidate_pairs":int(len(fa)*len(fb)),"queries":len(fa),"references":len(fb),
             "nearest_mean":float(scores.mean()),"nearest_std":float(scores.std(ddof=1)),
             "nearest_p95":float(np.quantile(scores,.95)),"nearest_p99":float(np.quantile(scores,.99)),"nearest_max":float(scores.max()),
             "query_images_with_nearest_at_or_above":{str(t):int((scores>=t).sum()) for t in thresholds},
             "pair_counts_at_or_above":counts}
    return summary,nearest,[{"comparison":f"{a}-{b}","cosine":s,"left":x,"right":y} for s,x,y in top]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--root",type=Path,required=True); ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--batch",type=int,default=64); ap.add_argument("--workers",type=int,default=4); a=ap.parse_args()
    device="cuda" if torch.cuda.is_available() else "cpu"; print("device",device,flush=True)
    a.out.mkdir(parents=True,exist_ok=True); cache=a.out/"resnet18_embeddings.pt"
    if cache.exists():
        saved=torch.load(cache,map_location="cpu",weights_only=False); f=saved["features"].float(); r=np.asarray(saved["roles"]); p=np.asarray(saved["paths"])
        print("reused cached embeddings",flush=True)
    else:
        f,r,p=extract(a.root,a.batch,a.workers,device)
        torch.save({"features":f.half(),"roles":r.tolist(),"paths":p.tolist()},cache)
    summaries=[]; nearest=[]; top=[]
    for left,right in (("validation","train"),("test","train"),("validation","test")):
        li=np.where(r==left)[0]; ri=np.where(r==right)[0]
        s,n,t=compare(left,right,f[li],f[ri],p[li],p[ri]); summaries.append(s); nearest+=n; top+=t
    with (a.out/"nearest_cross_split.csv").open("w",newline="",encoding="utf-8") as h:
        w=csv.DictWriter(h,fieldnames=nearest[0]); w.writeheader(); w.writerows(nearest)
    with (a.out/"top_cross_split_pairs.csv").open("w",newline="",encoding="utf-8") as h:
        w=csv.DictWriter(h,fieldnames=top[0]); w.writeheader(); w.writerows(sorted(top,key=lambda x:x["cosine"],reverse=True))
    hashes={k:{} for k in ROLES}
    for role,path in zip(r,p):
        digest=hashlib.sha256(Path(path).read_bytes()).hexdigest(); hashes[role].setdefault(digest,[]).append(path)
    exact={}
    for left,right in (("validation","train"),("test","train"),("validation","test")):
        shared=set(hashes[left])&set(hashes[right]); exact[f"{left}-{right}"]={"shared_sha256":len(shared),"left_images":sum(len(hashes[left][h]) for h in shared),"right_images":sum(len(hashes[right][h]) for h in shared)}
    (a.out/"summary.json").write_text(json.dumps({"extractor":"torchvision ResNet-18 ImageNet-1K V1, 512-D penultimate features","metric":"cosine similarity","device":device,"splits":{k:int((r==k).sum()) for k in ROLES},"exact_binary_duplicates":exact,"comparisons":summaries},indent=2),encoding="utf-8")
    print(json.dumps(summaries,indent=2),flush=True)
if __name__=="__main__": main()
