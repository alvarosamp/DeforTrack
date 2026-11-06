"""Training, checkpoint selection and evaluation of the 17 V2 models on Colab."""
import argparse
import csv
import gc
import hashlib
import json
import math
import os
import shutil
import sys
import time
import urllib.request
from pathlib import Path

os.environ.setdefault('KERAS_BACKEND', 'torch')
sys.path.insert(0, str(Path(__file__).resolve().parent))
MANIFEST_HASH = 'e14eecf7e8c253dfb1182fc31a916587148f0f3bd66ee21337d5c1b05a090b3c'
COUNTS = dict(train=12861, valid=833, test=892)
TRAINING_SEED = 0


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        raise ValueError('Empty output table')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def snapshot(work, target, state):
    pointer = target / 'current.json'
    old = read_json(pointer) if pointer.exists() else None
    slot = 1 - old['slot'] if old else 0
    destination = target / f'slot{slot}'
    files = {}
    for path in work.rglob('*'):
        if path.is_file() and path.suffix in {'.pt', '.pth', '.keras', '.csv', '.yaml', '.json'}:
            relative = path.relative_to(work)
            copy = destination / relative
            copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, copy)
            expected = sha(path)
            if sha(copy) != expected:
                raise ValueError(f'Drive copy mismatch: {relative}')
            files[str(relative)] = expected
    save_json(pointer, dict(slot=slot, files=files, state=state))
    return destination


def restore(work, target):
    if not (target / 'current.json').exists():
        return None
    pointer = read_json(target / 'current.json')
    source = target / f"slot{pointer['slot']}"
    for name, expected in pointer['files'].items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Invalid snapshot path')
        path = source / relative
        if sha(path) != expected:
            raise ValueError(f'Interrupted or changed Drive snapshot: {path}')
        destination = work / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    return pointer['state']


def prepare(args):
    from common.validation import validate_rows
    bundle, data = args.bundle, args.data
    manifest = bundle / 'partition_manifest.csv'
    if sha(manifest) != MANIFEST_HASH:
        raise ValueError('Wrong local derived V2 manifest')
    rows = read_csv(manifest)
    validate_rows(rows)
    for index, row in enumerate(rows, 1):
        for kind, key in [('images', 'image'), ('labels', 'label')]:
            if sha(data / row['split'] / kind / row[key]) != row[key + '_sha256']:
                raise ValueError(f'Dataset hash differs: {row[key]}')
        if index % 2000 == 0:
            print('Verified hashes', index, '/', len(rows), flush=True)
    work = args.work
    work.mkdir(parents=True, exist_ok=True)
    for split in COUNTS:
        paths = [str(data / split / 'images' / r['image']) for r in rows if r['split'] == split]
        (work / f'{split}.txt').write_text('\n'.join(paths) + '\n')
    import yaml
    (work / 'data.yaml').write_text(yaml.safe_dump(dict(path=str(data), train=str(work/'train.txt'),
                 val=str(work/'valid.txt'), test=str(work/'test.txt'), nc=1, names=['Forest'])))
    queue = read_json(bundle / 'new_training_queue.json')['jobs']
    if len(queue) != 17 or len({j['model'] for j in queue}) != 17:
        raise ValueError('Expected all 17 variants')
    for job in queue:
        if job['partition_manifest_sha256'] != MANIFEST_HASH or job['selection_images'] != 833:
            raise ValueError('Queue selection differs')
    return rows, queue


def official_weight(args, job):
    registry_name = 'weights_registry.json' if job['family'] == 'ultralytics' else 'maskrcnn_weights_registry.json'
    entry = read_json(args.bundle / registry_name)[job['model']]
    expected_prefix = 'https://github.com/ultralytics/assets/releases/download/' if job['family']=='ultralytics' else 'https://dl.fbaipublicfiles.com/detectron2/'
    if not entry['source_url'].startswith(expected_prefix) or entry['initialization_type']!='official COCO pretrained':
        raise ValueError('Invalid official weight provenance')
    if job['family']=='detectron2' and entry['config']!=job['config']:
        raise ValueError('Initialization architecture differs')
    target = args.output / 'official_initial_weights' / Path(entry['source_url']).name
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        temporary = target.with_suffix(target.suffix + '.download')
        request = urllib.request.Request(entry['source_url'], headers={'User-Agent': 'DeforTrack-audit'})
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open('wb') as f:
            shutil.copyfileobj(response, f)
        if sha(temporary) != entry['sha256']:
            raise ValueError('Official weight download hash differs')
        temporary.replace(target)
    if sha(target) != entry['sha256']:
        raise ValueError('Changed official initial weights')
    return target, entry


def finish(args, job, work, target, best, state, parameters):
    directory = snapshot(work, target, state)
    selected = directory / best.relative_to(work)
    import torch
    save_json(target / 'completed.json', dict(model=job['model'], family=job['family'],
              completed=True, best_checkpoint=str(selected), best_sha256=sha(selected),
              partition_manifest_sha256=MANIFEST_HASH, num_classes=1,
              final_training_gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
              selection_split='valid', selection_images=833, test_used_for_selection=False,
              state=state, parameters=parameters))


def train_yolo(args, job, work, target, state):
    from ultralytics import YOLO
    initial, provenance = official_weight(args, job)
    parameters = dict(data=str(args.work/'data.yaml'), epochs=2000, patience=5,
                      imgsz=640, batch=2, nbs=64, optimizer='auto', seed=TRAINING_SEED, deterministic=True,
                      amp=True, cache=False, device=0, workers=2, save=True, plots=False,
                      save_dir=str(work), project=str(work.parent), name=work.name, exist_ok=True)
    save_json(work/'protocol.json', dict(parameters=parameters, initialization=provenance))
    model = YOLO(str(work/'weights/last.pt') if state else str(initial))
    if state and (state['epoch'] >= 2000 or state['epoch'] - state['best_epoch'] >= 5):
        finish(args,job,work,target,work/'weights/best.pt',state,parameters)
        return
    if not state and len(model.names) != 80:
        raise ValueError('Expected untouched COCO 80-class initialization')
    def start(trainer):
        if Path(trainer.save_dir).resolve() != work.resolve():
            raise ValueError('Output directory changed')
        if len(trainer.train_loader.dataset)!=12861 or len(trainer.test_loader.dataset)!=833:
            raise ValueError('YOLO loader omitted images or loaded another validation partition')
        if state:
            trainer.stopper.best_epoch = state['best_epoch']
            trainer.stopper.best_fitness = state['best_fitness']
            trainer.stopper.possible_stop = state['epoch'] - state['best_epoch'] >= 4
    def saved(trainer):
        if trainer.args.batch!=2 or trainer.args.imgsz!=640 or trainer.args.nbs!=64:
            raise ValueError('Training batch/resolution changed; do not silently accept an OOM fallback')
        snapshot(work, target, dict(epoch=trainer.epoch+1, best_epoch=trainer.stopper.best_epoch,
                 best_fitness=float(trainer.best_fitness)))
    model.add_callback('on_train_start', start)
    model.add_callback('on_model_save', saved)
    if state:
        model.train(resume=True, data=parameters['data'], device=0, save_dir=str(work))
    else:
        model.train(**parameters)
    trainer = model.trainer
    finish(args, job, work, target, Path(trainer.best),
           dict(epoch=trainer.epoch+1, best_epoch=trainer.stopper.best_epoch,
                best_fitness=float(trainer.best_fitness)), parameters)


def train_unet(args, job, work, target, state, rows):
    import cv2
    import keras
    import numpy as np
    from mask_rcnn.data import instance_masks
    from u_net.model import build_unet
    keras.utils.set_random_seed(TRAINING_SEED)
    keras.mixed_precision.set_global_policy('mixed_float16')
    class Dataset(keras.utils.PyDataset):
        def __init__(self, role):
            super().__init__(workers=1, use_multiprocessing=False, max_queue_size=2)
            self.role = role
            self.rows = [r for r in rows if r['split'] == role]
            self.ids = np.arange(len(self.rows))
            self.rng = np.random.default_rng(TRAINING_SEED)
            if role == 'train' and state:
                self.ids = np.asarray(state['next_train_indices'])
                self.rng.bit_generator.state = state['rng']
        def __len__(self):
            return math.ceil(len(self.rows)/2)
        def __getitem__(self, index):
            images, masks = [], []
            for item in self.ids[index*2:(index+1)*2]:
                row = self.rows[item]
                image = cv2.imread(str(args.data/self.role/'images'/row['image']))
                if image is None or image.shape[:2] != (640,640):
                    raise ValueError('Unreadable image')
                reference = np.zeros((640,640), dtype=np.uint8)
                for instance in instance_masks((args.data/self.role/'labels'/row['label']).read_text(),640,640):
                    reference |= instance
                images.append(cv2.cvtColor(image,cv2.COLOR_BGR2RGB).astype(np.float32)/255)
                masks.append(reference[...,None].astype(np.float32))
            return np.stack(images),np.stack(masks)
        def on_epoch_end(self):
            if self.role == 'train': self.rng.shuffle(self.ids)
    train, valid = Dataset('train'), Dataset('valid')
    if state:
        model = keras.models.load_model(work/'last.keras')
    else:
        model = build_unet(640)
        model.compile(optimizer=keras.optimizers.Adam(0.001), loss='binary_crossentropy', jit_compile=False)
    current = dict(state or dict(epoch=0, best_loss=None, bad_epochs=0))
    if state and (current['bad_epochs'] >= 5 or current['epoch'] >= 2000):
        finish(args,job,work,target,work/'best.keras',current,
               dict(imgsz=640,batch=2,epochs=2000,patience=5,seed=TRAINING_SEED,Adam_lr=0.001,loss='BCE',initialization='new random',selection='valid833 val_loss'))
        return
    class Persist(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            value = float(logs['val_loss'])
            if not math.isfinite(value): raise ValueError('Nonfinite validation loss')
            if current['best_loss'] is None or value < current['best_loss']:
                current.update(best_loss=value,bad_epochs=0,best_epoch=epoch+1)
                self.model.save(work/'best.keras')
            else: current['bad_epochs'] += 1
            current.update(epoch=epoch+1,next_train_indices=train.ids.tolist(),rng=train.rng.bit_generator.state)
            self.model.save(work/'last.keras')
            snapshot(work,target,current)
            if current['bad_epochs'] >= 5: self.model.stop_training=True
    model.fit(train,validation_data=valid,initial_epoch=current['epoch'],epochs=2000,
              callbacks=[keras.callbacks.CSVLogger(str(work/'history.csv'),append=bool(state)),Persist()],verbose=2)
    finish(args,job,work,target,work/'best.keras',current,
           dict(imgsz=640,batch=2,epochs=2000,patience=5,seed=TRAINING_SEED,Adam_lr=0.001,loss='BCE',initialization='new random',selection='valid833 val_loss'))


class ValidationStop(Exception):
    pass


def train_mask(args, job, work, target, state, records):
    from detectron2 import model_zoo
    from detectron2.config import get_cfg
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from detectron2.engine import DefaultTrainer, HookBase
    from detectron2.evaluation import COCOEvaluator
    from detectron2.utils.env import seed_all_rng
    initial, provenance = official_weight(args,job)
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(job['config']))
    # Same number of images per schedule as the batch-2 reference run.
    scale = cfg.SOLVER.IMS_PER_BATCH/2
    cfg.SOLVER.MAX_ITER = int(cfg.SOLVER.MAX_ITER*scale)
    cfg.SOLVER.STEPS = tuple(int(s*scale) for s in cfg.SOLVER.STEPS)
    cfg.SOLVER.WARMUP_ITERS = int(cfg.SOLVER.WARMUP_ITERS*scale)
    cfg.SOLVER.BASE_LR *= 2/cfg.SOLVER.IMS_PER_BATCH
    cfg.SOLVER.IMS_PER_BATCH=2
    cfg.SOLVER.AMP.ENABLED=True
    cfg.MODEL.WEIGHTS=str(initial)
    cfg.MODEL.DEVICE='cuda'
    cfg.MODEL.ROI_HEADS.NUM_CLASSES=1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST=0.001
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST=0.5
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST=1000
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST=1000
    cfg.TEST.DETECTIONS_PER_IMAGE=100
    cfg.INPUT.MIN_SIZE_TRAIN=(640,)
    cfg.INPUT.MAX_SIZE_TRAIN=640
    cfg.INPUT.MIN_SIZE_TEST=640
    cfg.INPUT.MAX_SIZE_TEST=640
    cfg.INPUT.MASK_FORMAT='bitmask'
    cfg.TEST.EVAL_PERIOD=math.ceil(COUNTS['train']/2)
    cfg.SOLVER.CHECKPOINT_PERIOD=cfg.SOLVER.MAX_ITER+1
    cfg.DATALOADER.NUM_WORKERS=2
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS=False
    cfg.SEED=TRAINING_SEED
    cfg.OUTPUT_DIR=str(work)
    for role in ('train','valid'):
        name=job['model']+'_'+role
        if name in DatasetCatalog.list(): DatasetCatalog.remove(name)
        DatasetCatalog.register(name,lambda role=role:records[role])
        MetadataCatalog.get(name).set(thing_classes=['Forest'])
    cfg.DATASETS.TRAIN=(job['model']+'_train',)
    cfg.DATASETS.TEST=(job['model']+'_valid',)
    (work/'config.yaml').write_text(cfg.dump())
    current=dict(state or dict(step=0,best_ap=None,bad_validations=0))
    parameters=dict(schedule='official model-zoo schedule scaled for batch2; new protocol',
                    max_iterations=cfg.SOLVER.MAX_ITER,steps=list(cfg.SOLVER.STEPS),lr=cfg.SOLVER.BASE_LR,
                    evaluation_period=cfg.TEST.EVAL_PERIOD,patience_validations=5,seed=cfg.SEED,initialization=provenance)
    if state and (current['bad_validations'] >= 5 or current['step'] >= cfg.SOLVER.MAX_ITER):
        finish(args,job,work,target,work/'best_validation.pth',current,parameters)
        return
    class Select(HookBase):
        def select(self):
            value, recorded_step=self.trainer.storage.latest()['segm/AP']
            if recorded_step != self.trainer.iter: raise ValueError('Stale validation AP')
            if not math.isfinite(value): raise ValueError('Nonfinite validation AP')
            if current['best_ap'] is None or value > current['best_ap']:
                current.update(best_ap=float(value),best_step=min(self.trainer.iter+1,cfg.SOLVER.MAX_ITER),bad_validations=0)
                self.trainer.checkpointer.save('best_validation')
            else: current['bad_validations']+=1
            current['step']=min(self.trainer.iter+1,cfg.SOLVER.MAX_ITER)
            self.trainer.checkpointer.save('latest')
            snapshot(work,target,current)
        def after_step(self):
            step=self.trainer.iter+1
            if step < cfg.SOLVER.MAX_ITER and step % cfg.TEST.EVAL_PERIOD == 0:
                self.select()
                if current['bad_validations'] >= 5: raise ValidationStop()
        def after_train(self):
            if self.trainer.iter >= cfg.SOLVER.MAX_ITER: self.select()
    class Trainer(DefaultTrainer):
        @classmethod
        def build_evaluator(cls,config,name,output_folder=None):
            if name != job['model']+'_valid': raise ValueError('Test cannot select weights')
            return COCOEvaluator(name,tasks=('segm',),distributed=False,use_fast_impl=False,output_dir=str(work/'valid_eval'))
        def build_hooks(self):
            return super().build_hooks()+[Select()]
    seed_all_rng(cfg.SEED)
    trainer=Trainer(cfg)
    if state:
        (work/'last_checkpoint').write_text('latest.pth')
    trainer.resume_or_load(resume=bool(state))
    try: trainer.train()
    except ValidationStop: pass
    finish(args,job,work,target,work/'best_validation.pth',current,parameters)


def predictor(job, record):
    import cv2
    import numpy as np
    import torch
    best=record['best_checkpoint']
    family=job['family']
    if family=='ultralytics':
        from ultralytics import YOLO
        model=YOLO(best)
        def infer(image,score):
            output=model.predict(image,imgsz=640,conf=score,iou=0.7,max_det=100,device=0,verbose=False)[0]
            masks=[] if output.masks is None else output.masks.data.detach().cpu().numpy()
            masks=[cv2.resize(m.astype(np.uint8),(image.shape[1],image.shape[0]),interpolation=cv2.INTER_NEAREST)>0 for m in masks]
            scores=[] if output.boxes is None else output.boxes.conf.detach().cpu().numpy().tolist()
            return masks,scores
    elif family=='detectron2':
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor
        cfg=get_cfg(); cfg.merge_from_file(str(Path(best).parents[0]/'config.yaml'))
        cfg.MODEL.WEIGHTS=best;cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST=0.001
        model=DefaultPredictor(cfg)
        def infer(image,score):
            output=model(image)['instances'].to('cpu')
            keep=output.scores.numpy()>=score
            return output.pred_masks.numpy()[keep],output.scores.numpy()[keep].tolist()
    else:
        import keras
        model=keras.models.load_model(best,compile=False)
        def infer(image,score):
            resized=cv2.resize(image,(640,640),interpolation=cv2.INTER_LINEAR)
            batch=cv2.cvtColor(resized,cv2.COLOR_BGR2RGB).astype(np.float32)[None]/255
            prediction=np.asarray(model.predict(batch,verbose=0))[0,...,0]>0.5
            if prediction.shape != image.shape[:2]:
                prediction=cv2.resize(prediction.astype(np.uint8),(image.shape[1],image.shape[0]),interpolation=cv2.INTER_NEAREST)>0
            return [prediction], [1.0]
    return infer


def test_ground_truth(rows,data):
    import cv2
    from pycocotools import mask as mask_utils
    from mask_rcnn.data import instance_masks
    result=dict(info={},images=[],annotations=[],categories=[dict(id=1,name='Forest')])
    for image_id,row in enumerate(rows,1):
        image=cv2.imread(str(data/'test/images'/row['image']))
        if image is None: raise ValueError('Unreadable test image')
        h,w=image.shape[:2]
        result['images'].append(dict(id=image_id,file_name=row['image'],height=h,width=w))
        for mask in instance_masks((data/'test/labels'/row['label']).read_text(),h,w):
            rle=mask_utils.encode(__import__('numpy').asfortranarray(mask));rle['counts']=rle['counts'].decode()
            result['annotations'].append(dict(id=len(result['annotations'])+1,image_id=image_id,category_id=1,
                    segmentation=rle,area=float(mask.sum()),bbox=mask_utils.toBbox(rle).tolist(),iscrowd=0))
    return result


def evaluate(args,job,record,rows):
    import cv2
    import numpy as np
    import torch
    from pycocotools import mask as mask_utils
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from yolov8_yolov11.evaluate import metrics,yolo_mask
    gpu_key=''.join(c if c.isalnum() else '_' for c in torch.cuda.get_device_name(0))
    target=args.output/'evaluation'/gpu_key/job['model'];target.mkdir(parents=True,exist_ok=True)
    if (target/'complete.json').exists():
        previous=read_json(target/'complete.json')
        if previous['weights_sha256'] != record['best_sha256']: raise ValueError('Evaluation weights changed')
        for name,h in previous['files'].items():
            if sha(target/name)!=h: raise ValueError('Evaluation result changed')
        return read_json(target/'summary.json')
    if sha(record['best_checkpoint']) != record['best_sha256']: raise ValueError('Changed selected checkpoint')
    infer=predictor(job,record)
    test=[r for r in rows if r['split']=='test']
    datasets=[('test',args.data/'test/images',None,test)]
    for name,count in [('ForestAerial',5022),('AmazonAtlantic',599)]:
        base=args.external/name
        metadata={r['image']:r['mask'] for r in read_csv(base/'meta_data.csv')}
        if name=='ForestAerial':
            keep={r['image'] for r in read_csv(args.bundle/'forest_aerial_overlap.csv') if r['included_in_screened_subset'].lower()=='true'}
            metadata={image:mask for image,mask in metadata.items() if image in keep}
        if len(metadata)!=count: raise ValueError(f'Wrong external subset {name}')
        datasets.append((name,base/'images',base/'masks',[dict(image=i,label=m) for i,m in sorted(metadata.items())]))
    ap_detections=[]; summaries=[]
    for dataset,images,masks,items in datasets:
        ledger=[]
        for image_id,row in enumerate(items,1):
            image=cv2.imread(str(images/row['image']))
            if image is None: raise ValueError('Unreadable image')
            if dataset=='test': reference=yolo_mask(args.data/'test/labels'/row['label'],*image.shape[:2])
            else:
                gray=cv2.imread(str(masks/row['label']),cv2.IMREAD_GRAYSCALE)
                if gray is None or gray.shape!=image.shape[:2]: raise ValueError('Invalid external mask')
                reference=gray>127
            instances,scores=infer(image,0.001)
            prediction=np.zeros(image.shape[:2],dtype=bool)
            threshold=0.30 if job['family']=='detectron2' else 0.25
            for instance,score in zip(instances,scores):
                if score>=threshold: prediction |= instance
                if dataset=='test' and job['family']!='keras':
                    rle=mask_utils.encode(np.asfortranarray(instance.astype(np.uint8)));rle['counts']=rle['counts'].decode()
                    ap_detections.append(dict(image_id=image_id,category_id=1,segmentation=rle,score=float(score)))
            ledger.append(dict(model=job['model'],image=row['image'],**metrics(prediction,reference)))
            if image_id%100==0: print(job['model'],dataset,image_id,'/',len(items),flush=True)
        write_csv(target/f'{dataset}_per_image.csv',ledger)
        summary=dict(dataset=dataset,n=len(ledger),model=job['model'])
        for key in list(ledger[0])[2:]:
            values=np.asarray([r[key] for r in ledger])
            summary.update({key+'_mean':float(values.mean()),key+'_std':float(values.std(ddof=1)),
                            key+'_ci95_halfwidth':float(1.96*values.std(ddof=1)/np.sqrt(len(values)))})
        summaries.append(summary)
    ap=None
    if job['family']!='keras':
        gt=COCO();gt.dataset=test_ground_truth(test,args.data);gt.createIndex()
        if ap_detections: detections=gt.loadRes(ap_detections)
        else:
            detections=COCO();detections.dataset=dict(images=gt.dataset['images'],categories=gt.dataset['categories'],annotations=[]);detections.createIndex()
        evaluator=COCOeval(gt,detections,'segm');evaluator.params.maxDets=[1,10,100]
        evaluator.evaluate();evaluator.accumulate();evaluator.summarize()
        ap=dict(AP=float(evaluator.stats[0]),AP50=float(evaluator.stats[1]),AP75=float(evaluator.stats[2]),
                confidence=0.001,max_detections=100,implementation='pycocotools.COCOeval')
    del infer
    if job['family']=='keras': __import__('keras').backend.clear_session()
    gc.collect();torch.cuda.empty_cache()
    import subprocess
    subprocess.run([sys.executable,str(Path(__file__).with_name('profile_checkpoint.py')),
        '--bundle',str(args.bundle),'--data',str(args.data),
        '--record',str(args.output/'models'/job['model']/'completed.json'),
        '--output',str(target/'computational_profile.json')],check=True)
    profile=read_json(target/'computational_profile.json')
    summary=dict(model=job['model'],weights_sha256=record['best_sha256'],region=summaries,common_ap=ap,profile=profile)
    save_json(target/'summary.json',summary)
    files={p.name:sha(p) for p in target.iterdir() if p.is_file() and p.name!='complete.json'}
    save_json(target/'complete.json',dict(weights_sha256=record['best_sha256'],files=files))
    return summary


def main():
    parser=argparse.ArgumentParser()
    for key in ('bundle','data','external','output','work'): parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--preflight-only',action='store_true')
    parser.add_argument('--recover-interrupted-lease',action='store_true')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    import torch,ultralytics
    if not torch.cuda.is_available(): raise ValueError('GPU required; no CPU fallback')
    environment_matrix={'2.8.0':{'12.6','12.8','12.9'},'2.11.0':{'13.0'}}
    if torch.version.cuda not in environment_matrix.get(torch.__version__.split('+')[0],set()) or ultralytics.__version__!='8.4.96':
        raise ValueError('Wrong environment versions')
    rows,queue=prepare(args)
    for name in ('ForestAerial','AmazonAtlantic'):
        metadata=read_csv(args.external/name/'meta_data.csv')
        if {row['image']:row['mask'] for row in metadata} != {row['image']:row['mask'] for row in read_csv(args.bundle/(name+'_meta_data.csv'))}:
            raise ValueError('External metadata differs from the audited source')
        keep=None
        if name=='ForestAerial':
            keep={r['image'] for r in read_csv(args.bundle/'forest_aerial_overlap.csv') if r['included_in_screened_subset'].lower()=='true'}
        selected=[row for row in metadata if keep is None or row['image'] in keep]
        if len(selected)!=(5022 if name=='ForestAerial' else 599): raise ValueError('External subset count differs')
        content=[]
        for row in metadata:
            for kind,key in [('images','image'),('masks','mask')]:
                if not (args.external/name/kind/row[key]).is_file(): raise ValueError('External dataset incomplete')
        for row in selected:
            content.append(dict(image=row['image'],image_sha256=sha(args.external/name/'images'/row['image']),
                                mask=row['mask'],mask_sha256=sha(args.external/name/'masks'/row['mask'])))
        saved=args.output/(name+'_content_manifest.json')
        if saved.exists() and read_json(saved)!=content: raise ValueError('External files changed across Colab sessions')
        save_json(saved,content)
    code_root=Path(__file__).resolve().parent
    code_paths=sorted(code_root.rglob('*.py'))
    protocol=dict(counts=COUNTS,partition_manifest_sha256=MANIFEST_HASH,models=17,training_seed=TRAINING_SEED,
          gpu=torch.cuda.get_device_name(0),torch=torch.__version__,ultralytics=ultralytics.__version__,
          selection='valid833 only',test_used_for_selection=False,pixel_max_detections=100,
          boundary='3 erosion iterations 3x3; dilation elliptic7x7; tolerance3px',
          scope='new local V2 derivative; not historical results or published V2 counts',
          script_sha256=sha(__file__),code_sha256={p.relative_to(code_root).as_posix():sha(p) for p in code_paths})
    if (args.output/'protocol.json').exists():
        previous_protocol=read_json(args.output/'protocol.json')
        if previous_protocol.get('torch')!=protocol['torch'] or previous_protocol.get('ultralytics')!=protocol['ultralytics']:
            raise ValueError('Training framework changed across sessions; do not mix environments in this round')
        if previous_protocol.get('code_sha256')!=protocol['code_sha256']:
            raise ValueError('Code changed during existing round; use a separate output directory')
    save_json(args.output/'protocol.json',protocol)
    history=args.output/'session_history.json'
    sessions=read_json(history) if history.exists() else []
    sessions.append(dict(started_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),gpu=protocol['gpu'],torch=protocol['torch']))
    save_json(history,sessions)
    print(json.dumps(protocol,indent=2),flush=True)
    if args.preflight_only: return
    lease=args.output/'active_session.json'
    if lease.exists():
        if not args.recover_interrupted_lease:
            raise ValueError('Previous session owns Drive lease. Confirm it is stopped before using --recover-interrupted-lease.')
        abandoned=args.output/'abandoned_sessions.json'
        previous=read_json(abandoned) if abandoned.exists() else []
        previous.append(read_json(lease))
        save_json(abandoned,previous)
        lease.unlink()
    token=__import__('uuid').uuid4().hex
    owner=dict(token=token,pid=os.getpid(),gpu=protocol['gpu'])
    with lease.open('x') as stream: json.dump(owner,stream)
    records=None
    for job in queue:
        target=args.output/'models'/job['model'];target.mkdir(parents=True,exist_ok=True)
        if (target/'completed.json').exists():
            record=read_json(target/'completed.json')
            if sha(record['best_checkpoint'])!=record['best_sha256']: raise ValueError('Completed model changed')
            continue
        work=args.work/job['model'];work.mkdir(parents=True,exist_ok=True)
        state=restore(work,target)
        save_json(args.output/'status.json',dict(phase='training',model=job['model'],resuming=bool(state)))
        if job['family']=='ultralytics': train_yolo(args,job,work,target,state)
        elif job['family']=='keras': train_unet(args,job,work,target,state,rows)
        elif job['family']=='detectron2':
            if records is None:
                from mask_rcnn.data import build_records
                records={role:build_records(rows,role,args.data,COUNTS[role]) for role in ('train','valid')}
            train_mask(args,job,work,target,state,records)
        else: raise ValueError('Unknown family')
        gc.collect();torch.cuda.empty_cache()
        if job['family']=='keras': __import__('keras').backend.clear_session()
    summaries=[]
    for job in queue:
        save_json(args.output/'status.json',dict(phase='evaluating',model=job['model']))
        record=read_json(args.output/'models'/job['model']/'completed.json')
        summaries.append(evaluate(args,job,record,rows))
        gc.collect();torch.cuda.empty_cache()
    if len(summaries)!=17 or sum(s['common_ap'] is not None for s in summaries)!=16: raise ValueError('Incomplete results')
    save_json(args.output/'all_results.json',summaries)
    regions=[r for s in summaries for r in s['region']]
    write_csv(args.output/'region_all.csv',regions)
    write_csv(args.output/'common_ap_16.csv',[dict(model=s['model'],**s['common_ap']) for s in summaries if s['common_ap'] is not None])
    write_csv(args.output/'computational_profiles_17.csv',[dict(model=s['model'],**s['profile']) for s in summaries])
    rankings=[]
    for dataset in ('test','ForestAerial','AmazonAtlantic'):
        ordered=sorted([r for r in regions if r['dataset']==dataset],key=lambda r:r['iou_mean'],reverse=True)
        rankings.extend(dict(rank=i,**r) for i,r in enumerate(ordered,1))
    write_csv(args.output/'rankings.csv',rankings)
    save_json(args.output/'status.json',dict(phase='complete',models=17,common_ap_models=16,article_changed=False,
        note='New results require scientific/manuscript review; no geographic or temporal independence claim'))
    import zipfile
    with zipfile.ZipFile(args.output/'public_supplement.zip','w',compression=zipfile.ZIP_DEFLATED,allowZip64=True) as archive:
        for path in args.output.iterdir():
            if path.is_file() and path.suffix in {'.csv','.json'}: archive.write(path,path.name)
        for path in (args.output/'evaluation').rglob('*'):
            if path.is_file() and path.suffix in {'.csv','.json'}: archive.write(path,path.relative_to(args.output))
        for path in args.bundle.iterdir():
            if path.is_file() and path.suffix in {'.csv','.json'}: archive.write(path,Path('protocol_inputs')/path.name)
        for path in code_paths: archive.write(path,Path('code')/path.relative_to(code_root))
    print('COMPLETE. Results and public_supplement.zip saved on Drive. No raw weights/images in supplement.',flush=True)
    if read_json(lease)==owner: lease.unlink()


if __name__=='__main__':
    main()
