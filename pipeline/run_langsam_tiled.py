from repository_paths import image_path
import argparse,hashlib,json,time,traceback
from datetime import datetime,timezone
from dataclasses import asdict,is_dataclass
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
from lang_sam import LangSAM

p=argparse.ArgumentParser()
p.add_argument('--baseline',required=True)
p.add_argument('--output',required=True)
p.add_argument('--tile-size',type=int,default=1024)
p.add_argument('--overlap',type=int,default=256)
a=p.parse_args()
base=json.loads(Path(a.baseline).read_text())
assert base['status']=='INFERENCE_COMPLETE'
source=image_path(base['image'])
assert hashlib.sha256(source.read_bytes()).hexdigest()==base['sha256']
assert 0<=a.overlap<a.tile_size
image=Image.open(source).convert('RGB')
w,h=image.size
out=Path(a.output).resolve()
out.mkdir(parents=True,exist_ok=False)
(out/'tiles').mkdir()
stride=a.tile_size-a.overlap

def origins(length):
 last=max(0,length-a.tile_size)
 positions=list(range(0,last+1,stride))
 if positions[-1]!=last: positions.append(last)
 assert positions[0]==0 and positions[-1]+a.tile_size>=length
 assert all(y-x<=a.tile_size for x,y in zip(positions,positions[1:]))
 return positions

xs,ys=origins(w),origins(h)
regions=[(x,y,min(x+a.tile_size,w),min(y+a.tile_size,h)) for y in ys for x in xs]
report={'status':'STARTED','started_at_utc':datetime.now(timezone.utc).isoformat(),'image':str(source),'sha256':base['sha256'],'original_size':[w,h],'mode':'fixed_overlapping_tiles','tile_size':a.tile_size,'minimum_overlap':a.overlap,'stride':stride,'tile_count':len(regions),'prompt':base['prompt'],'box_threshold':base['box_threshold'],'text_threshold':base['text_threshold'],'sam_type':base['sam_type'],'grounding_dino':base['grounding_dino'],'versions':base['versions'],'langsam_commit':base.get('langsam_commit'),'baseline_result':str(Path(a.baseline).resolve()),'ground_truth_used_for_inference':False,'accuracy_evaluated':False,'candidate_count_includes_tile_duplicates':True,'tiles':[],'candidates':[]}
(out/'result.json').write_text(json.dumps(report,indent=2))
font=ImageFont.load_default(size=28)
grid=image.copy()
gdraw=ImageDraw.Draw(grid)
for i,region in enumerate(regions,1):
 x1,y1,x2,y2=region
 gdraw.rectangle((x1,y1,x2-1,y2-1),outline='cyan',width=4)
 gdraw.text((x1+8,y1+8),'T%02d'%i,fill='cyan',font=font,stroke_width=2,stroke_fill='black')
grid.thumbnail((1600,1000))
grid.save(out/'grid_preview.jpg',quality=94)
try:
 assert torch.cuda.is_available()
 started=time.perf_counter()
 model=LangSAM(sam_type=report['sam_type'])
 torch.cuda.synchronize()
 report['model_load_seconds']=time.perf_counter()-started
 size=model.gdino.processor.image_processor.size
 report['detector_resize_config']=asdict(size) if is_dataclass(size) else size
 torch.cuda.reset_peak_memory_stats()
 union=np.zeros((h,w),dtype=bool)
 total_infer=0.0
 overall=time.perf_counter()
 for ti,(x1,y1,x2,y2) in enumerate(regions,1):
  tile_id='T%02d'%ti
  td=out/'tiles'/tile_id
  td.mkdir()
  crop=image.crop((x1,y1,x2,y2))
  cw,ch=crop.size
  crop.save(td/'input.jpg',quality=95)
  torch.cuda.synchronize()
  started=time.perf_counter()
  result=model.predict([crop],[report['prompt']],box_threshold=report['box_threshold'],text_threshold=report['text_threshold'])[0]
  torch.cuda.synchronize()
  elapsed=time.perf_counter()-started
  total_infer+=elapsed
  boxes=np.asarray(result['boxes'],dtype=np.float32).reshape(-1,4)
  count=len(boxes)
  masks=np.asarray(result['masks'],dtype=bool).reshape(count,ch,cw)
  scores=np.asarray(result['scores'],dtype=np.float32).reshape(-1)
  mask_scores=np.asarray(result['mask_scores'],dtype=np.float32).reshape(-1)
  assert count==len(scores)==len(mask_scores)
  np.savez_compressed(td/'prediction.npz',boxes=boxes,scores=scores,mask_scores=mask_scores,masks=masks)
  overlay=np.asarray(crop).copy()
  tile_candidates=[]
  for mi,(box,mask,score,mscore) in enumerate(zip(boxes,masks,scores,mask_scores),1):
   cid='C%04d'%(len(report['candidates'])+1)
   union[y1:y2,x1:x2]|=mask
   overlay[mask]=(overlay[mask].astype(np.float32)*0.55+np.asarray([255,50,50])*0.45).astype(np.uint8)
   global_box=(box+np.asarray([x1,y1,x1,y1])).tolist()
   candidate={'id':cid,'tile_id':tile_id,'tile_candidate_index':mi,'box_xyxy_local':box.tolist(),'box_xyxy_original':global_box,'detector_score':float(score),'sam_mask_score':float(mscore),'mask_area_pixels':int(mask.sum()),'mask_source':str((td/'prediction.npz').relative_to(out)),'mask_array_index':mi-1,'mask_origin_xy':[x1,y1]}
   report['candidates'].append(candidate)
   tile_candidates.append(cid)
  tile_overlay=Image.fromarray(overlay)
  draw=ImageDraw.Draw(tile_overlay)
  for box,cid in zip(boxes,tile_candidates):
   draw.rectangle(tuple(map(float,box)),outline='red',width=3)
   draw.text((max(0,float(box[0])),max(0,float(box[1])-30)),cid,fill='red',font=font,stroke_width=1,stroke_fill='black')
  tile_overlay.save(td/'overlay.jpg',quality=94)
  item={'tile_id':tile_id,'crop_xyxy_original':[x1,y1,x2,y2],'candidate_count':count,'candidate_ids':tile_candidates,'inference_seconds':elapsed}
  (td/'result.json').write_text(json.dumps(item,indent=2))
  report['tiles'].append(item)
  report['candidate_count_raw']=len(report['candidates'])
  (out/'result.json').write_text(json.dumps(report,indent=2,default=str))
  print('TILE',ti,'/',len(regions),'candidates',count,'seconds',round(elapsed,3),flush=True)
 report['inference_seconds_total']=total_infer
 report['processing_seconds_including_output']=time.perf_counter()-overall
 report['gpu_peak_allocated_gib']=torch.cuda.max_memory_allocated()/2**30
 report['gpu_peak_reserved_gib']=torch.cuda.max_memory_reserved()/2**30
 report['tiles_with_candidates']=sum(t['candidate_count']>0 for t in report['tiles'])
 report['union_mask_area_pixels']=int(union.sum())
 report['candidate_count_raw']=len(report['candidates'])
 Image.fromarray(union.astype(np.uint8)*255).save(out/'union_mask.png')
 final=np.asarray(image).copy()
 final[union]=(final[union].astype(np.float32)*0.55+np.asarray([255,50,50])*0.45).astype(np.uint8)
 final=Image.fromarray(final)
 draw=ImageDraw.Draw(final)
 draw.rectangle((0,0,w,48),fill='black')
 draw.text((10,6),'LangSAM tiles | candidates (with overlap duplicates): '+str(len(report['candidates'])),fill='white',font=font)
 final.save(out/'overlay_full.jpg',quality=95)
 final.thumbnail((1600,1000))
 final.save(out/'overlay_preview.jpg',quality=94)
 report.update(status='INFERENCE_COMPLETE',output_directory=str(out))
 (out/'result.json').write_text(json.dumps(report,indent=2,default=str))
 print(json.dumps({k:report[k] for k in ['status','tile_count','tiles_with_candidates','candidate_count_raw','union_mask_area_pixels','inference_seconds_total','gpu_peak_allocated_gib','output_directory','accuracy_evaluated']},indent=2),flush=True)
except Exception:
 report.update(status='FAILED',error=traceback.format_exc())
 (out/'result.json').write_text(json.dumps(report,indent=2,default=str))
 raise
