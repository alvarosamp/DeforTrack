"""Compare OR, AND and score-weighted mean fusion on Mask R-CNN outputs."""
import argparse, csv, json
from pathlib import Path
import cv2
import numpy as np
from evaluate_maskrcnn_defortrack_test import build_predictor, segmentation_metrics, yolo_mask

GRID = np.arange(0.05, 1.0, 0.05)
METRICS = ("iou", "dice_f1", "precision", "recall", "boundary_iou", "boundary_f1", "pixel_accuracy")

def instances(outputs, shape):
    x = outputs["instances"].to("cpu")
    if not len(x): return np.empty((0,*shape),bool), np.empty(0)
    keep = np.isin(x.pred_classes.numpy(), [0,1])
    masks, scores = x.pred_masks.numpy()[keep], x.scores.numpy()[keep]
    if len(masks) and masks.shape[1:] != shape:
        masks = np.stack([cv2.resize(m.astype("uint8"), shape[::-1], interpolation=cv2.INTER_NEAREST).astype(bool) for m in masks])
    return masks, scores

def predict_fusions(predictor, image, shape):
    masks, scores = instances(predictor(image), shape)
    if not len(masks):
        return np.zeros(shape,bool), np.zeros(shape,bool), np.zeros(shape,"float32"), 0
    weighted = np.tensordot(scores, masks.astype("float32"), axes=(0,0)) / scores.sum()
    return masks.any(0), masks.all(0), weighted, len(masks)

def validation_threshold(predictor, images, labels):
    values = {float(t): [] for t in GRID}
    paths = sorted(Path(images).glob("*"))
    for i,p in enumerate(paths,1):
        if p.suffix.lower() not in {".jpg",".jpeg",".png"}: continue
        im=cv2.imread(str(p)); gt=yolo_mask(Path(labels)/f"{p.stem}.txt",*im.shape[:2])
        _,_,weighted,_=predict_fusions(predictor,im,gt.shape)
        for t in GRID: values[float(t)].append(segmentation_metrics(weighted>=t,gt,3)["iou"])
        if i%100==0: print(f"validation {i}/{len(paths)}",flush=True)
    means={t:float(np.mean(v)) for t,v in values.items()}
    return max(means,key=means.get),means

def evaluate(predictor, images, labels, threshold):
    rows=[]; paths=sorted(Path(images).glob("*"))
    for i,p in enumerate(paths,1):
        if p.suffix.lower() not in {".jpg",".jpeg",".png"}: continue
        im=cv2.imread(str(p)); gt=yolo_mask(Path(labels)/f"{p.stem}.txt",*im.shape[:2])
        union,inter,weighted,n=predict_fusions(predictor,im,gt.shape)
        for method,pred in (("OR",union),("AND",inter),("weighted_mean",weighted>=threshold)):
            rows.append({"image":p.name,"instances":n,"method":method,**segmentation_metrics(pred,gt,3)})
        if i%100==0: print(f"test {i}/{len(paths)}",flush=True)
    return rows

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--validation",required=True); ap.add_argument("--test",required=True)
    ap.add_argument("--models-json",required=True); ap.add_argument("--model",default="mask_rcnn_R_101_FPN_3x"); ap.add_argument("--out",required=True)
    a=ap.parse_args(); models=json.loads(Path(a.models_json).read_text()); predictor=build_predictor(a.model,models[a.model],"cuda",.30,640)
    vimg=Path(a.validation)/"images"; vlbl=Path(a.validation)/"labels"; timg=Path(a.test)/"images"; tlbl=Path(a.test)/"labels"
    threshold,curve=validation_threshold(predictor,vimg,vlbl); rows=evaluate(predictor,timg,tlbl,threshold)
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    with (out/"fusion_per_image.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    summary=[]
    for method in ("OR","AND","weighted_mean"):
        part=[r for r in rows if r["method"]==method]; item={"method":method,"n":len(part),"mean_instances":float(np.mean([r["instances"] for r in part]))}
        for m in METRICS: item[m]=float(np.mean([r[m] for r in part])); item[m+"_std"]=float(np.std([r[m] for r in part],ddof=1))
        summary.append(item)
    (out/"fusion_summary.json").write_text(json.dumps({"model":a.model,"candidate_score_threshold":.30,"weighted_threshold":threshold,"validation_curve":curve,"test":summary},indent=2))
    print(json.dumps({"weighted_threshold":threshold,"test":summary},indent=2))
if __name__=="__main__": main()
