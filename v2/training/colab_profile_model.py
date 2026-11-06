"""Latency and memory profile of one checkpoint on the Colab GPU."""
import argparse
import gc
import json
import time
from pathlib import Path

from colab_full_revision import MANIFEST_HASH, predictor, read_csv, read_json, save_json, sha


def main():
    parser=argparse.ArgumentParser()
    for key in ('bundle','data','record','output'): parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    import cv2,numpy as np,torch
    from profile_model_local import ResourceSampler
    if not torch.cuda.is_available(): raise ValueError('CUDA required')
    if sha(args.bundle/'partition_manifest.csv') != MANIFEST_HASH: raise ValueError('Changed test manifest')
    record=read_json(args.record)
    if sha(record['best_checkpoint'])!=record['best_sha256']: raise ValueError('Changed checkpoint')
    job=next(j for j in read_json(args.bundle/'new_training_queue.json')['jobs'] if j['model']==record['model'])
    rows=[r for r in read_csv(args.bundle/'partition_manifest.csv') if r['split']=='test']
    if len(rows)!=892: raise ValueError('Wrong test count')
    paths=[]
    for row in rows:
        path=args.data/'test/images'/row['image']
        if sha(path)!=row['image_sha256']: raise ValueError('Changed test image')
        paths.append(path)
    infer=predictor(job,record)
    threshold=0.30 if job['family']=='detectron2' else 0.25
    for path in paths: infer(cv2.imread(str(path)),threshold)
    torch.cuda.synchronize()
    sampler=ResourceSampler();sampler.start();timings=[]
    for repetition in range(3):
        start=time.perf_counter()
        for path in paths: infer(cv2.imread(str(path)),threshold)
        torch.cuda.synchronize();timings.append((time.perf_counter()-start)/len(paths))
    sampler.stop();resources=sampler.summary(2676)
    resources={k:(v if not isinstance(v,float) or np.isfinite(v) else None) for k,v in resources.items()}
    latency=float(np.mean(timings))
    save_json(args.output,dict(gpu=torch.cuda.get_device_name(0),hardware='Colab, not RTX4060 reference',
        logical_cpus=sampler.logical_cpus,warmup=892,measured=2676,repetitions=3,
        latency_seconds=latency,fps=1/latency,latency_trial_std=float(np.std(timings,ddof=1)),
        checkpoint_size_mib=Path(record['best_checkpoint']).stat().st_size/(1024**2),
        weights_sha256=record['best_sha256'],test_image_names_sha256=__import__('hashlib').sha256(
            '\n'.join(sorted(r['image'] for r in rows)).encode()).hexdigest(),**resources))


if __name__=='__main__':main()
