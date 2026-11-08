import argparse,hashlib,json,time,traceback
from datetime import datetime,timezone
from pathlib import Path
import importlib.metadata as metadata
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
from lang_sam import LangSAM

parser=argparse.ArgumentParser()
parser.add_argument('--image',required=True)
parser.add_argument('--output',required=True)
parser.add_argument('--prompt',default='power line.')
parser.add_argument('--box-threshold',type=float,default=0.30)
parser.add_argument('--text-threshold',type=float,default=0.25)
args=parser.parse_args()
source=Path(args.image).resolve()
out=Path(args.output).resolve()
assert source.is_file(),source
out.mkdir(parents=True,exist_ok=False)
report={'status':'STARTED','started_at_utc':datetime.now(timezone.utc).isoformat(),'image':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'prompt':args.prompt,'box_threshold':args.box_threshold,'text_threshold':args.text_threshold,'sam_type':'sam2.1_hiera_small','grounding_dino':'IDEA-Research/grounding-dino-base','mode':'whole_image_baseline','ground_truth_used_for_inference':False,'versions':{name:metadata.version(name) for name in ['torch','torchvision','transformers','lang-sam','sam-2']}}
commit=Path('logs/langsam_source_commit.txt')
if commit.exists(): report['langsam_commit']=commit.read_text().strip()
(out/'result.json').write_text(json.dumps(report,indent=2))
try:
 assert torch.cuda.is_available(),'CUDA unavailable'
 image=Image.open(source).convert('RGB')
 width,height=image.size
 report['original_size']=[width,height]
 start=time.perf_counter()
 model=LangSAM(sam_type=report['sam_type'])
 torch.cuda.synchronize()
 report['model_load_seconds']=time.perf_counter()-start
 report['detector_resize_config']=model.gdino.processor.image_processor.size
 torch.cuda.reset_peak_memory_stats()
 torch.cuda.synchronize()
 start=time.perf_counter()
 result=model.predict([image],[args.prompt],box_threshold=args.box_threshold,text_threshold=args.text_threshold)[0]
 torch.cuda.synchronize()
 report['inference_seconds']=time.perf_counter()-start
 report['gpu_peak_allocated_gib']=torch.cuda.max_memory_allocated()/2**30
 report['gpu_peak_reserved_gib']=torch.cuda.max_memory_reserved()/2**30
 boxes=np.asarray(result['boxes'],dtype=np.float32).reshape(-1,4)
 count=len(boxes)
 masks=np.asarray(result['masks'],dtype=bool).reshape(count,height,width)
 scores=np.asarray(result['scores'],dtype=np.float32).reshape(-1)
 mask_scores=np.asarray(result['mask_scores'],dtype=np.float32).reshape(-1)
 assert len(scores)==len(mask_scores)==count
 np.savez_compressed(out/'prediction.npz',boxes=boxes,scores=scores,mask_scores=mask_scores,masks=masks)
 (out/'masks').mkdir()
 overlay=np.asarray(image).copy()
 palette=[(255,70,70),(30,220,80),(30,150,255),(255,180,20),(220,50,230),(20,230,220)]
 candidates=[]
 for i,(box,mask,score,mask_score) in enumerate(zip(boxes,masks,scores,mask_scores)):
  color=np.asarray(palette[i%len(palette)])
  overlay[mask]=(overlay[mask].astype(np.float32)*0.55+color*0.45).astype(np.uint8)
  maskfile='masks/%03d.png'%(i+1)
  Image.fromarray(mask.astype(np.uint8)*255).save(out/maskfile)
  candidates.append({'id':i+1,'box_xyxy':box.tolist(),'detector_score':float(score),'sam_mask_score':float(mask_score),'mask_area_pixels':int(mask.sum()),'mask':maskfile})
 display=Image.fromarray(overlay)
 draw=ImageDraw.Draw(display)
 font=ImageFont.load_default(size=28)
 for i,box in enumerate(boxes):
  x1,y1,x2,y2=map(float,box)
  color=palette[i%len(palette)]
  draw.rectangle([x1,y1,x2,y2],outline=color,width=5)
  draw.text((max(0,x1),max(0,y1-34)),str(i+1),fill=color,font=font,stroke_width=2,stroke_fill='black')
 caption=f'LangSAM whole image | prompt: {args.prompt} | candidates: {count}'
 draw.rectangle((0,0,width,48),fill='black')
 draw.text((12,7),caption,fill='white',font=font)
 display.save(out/'overlay_full.jpg',quality=95)
 preview=display.copy()
 preview.thumbnail((1600,1000))
 preview.save(out/'overlay_preview.jpg',quality=94)
 original=image.copy()
 original.thumbnail((1600,1000))
 original.save(out/'input_preview.jpg',quality=94)
 report.update(status='INFERENCE_COMPLETE',candidate_count=count,candidates=candidates,labels=[str(label) for label in result.get('labels',[])],mask_shape=list(masks.shape),output_directory=str(out),accuracy_evaluated=False)
 (out/'result.json').write_text(json.dumps(report,indent=2,default=str))
 print(json.dumps({key:report[key] for key in ['status','original_size','prompt','candidate_count','mask_shape','model_load_seconds','inference_seconds','gpu_peak_allocated_gib','detector_resize_config','output_directory','accuracy_evaluated']},indent=2),flush=True)
except Exception:
 report.update(status='FAILED',error=traceback.format_exc())
 (out/'result.json').write_text(json.dumps(report,indent=2,default=str))
 raise
