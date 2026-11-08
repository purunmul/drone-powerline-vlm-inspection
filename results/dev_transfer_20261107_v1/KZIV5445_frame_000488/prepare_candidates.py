from pathlib import Path
import json,math,hashlib,csv
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps
src=Path('results/dev_transfer_20261107_v1/KZIV5445_frame_000488/tiles')
out=Path('results/dev_transfer_20261107_v1/KZIV5445_frame_000488/candidates')
r=json.loads((src/'result.json').read_text())
assert r['status']=='INFERENCE_COMPLETE'
im=Image.open(r['image']).convert('RGB'); w,h=im.size
assert hashlib.sha256(Path(r['image']).read_bytes()).hexdigest()==r['sha256']
out.mkdir(parents=True,exist_ok=False)
cs=r['candidates']; masks=[]
for c in cs:
 with np.load(src/c['mask_source']) as z: m=z['masks'][c['mask_array_index']].astype(bool)
 x,y=c['mask_origin_xy']; mh,mw=m.shape
 assert x>=0 and y>=0 and x+mw<=w and y+mh<=h
 full=np.zeros((h,w),bool); full[y:y+mh,x:x+mw]=m
 assert int(full.sum())==c['mask_area_pixels']
 masks.append(full)
n=len(cs); areas=[int(m.sum()) for m in masks]; ious=np.eye(n)
for i in range(n):
 for j in range(i):
  inter=int((masks[i]&masks[j]).sum()); union=areas[i]+areas[j]-inter
  ious[i,j]=ious[j,i]=inter/union if union else 0
threshold=0.85; groups=[]
for i in sorted(range(n),key=lambda i:(-cs[i]['detector_score'],cs[i]['id'])):
 target=next((g for g in groups if all(ious[i,j]>=threshold for j in g)),None)
 if target is None: groups.append([i])
 else: target.append(i)
assert sorted(i for g in groups for i in g)==list(range(n))
font=ImageFont.load_default(size=24); records=[]; cards=[]; overview=im.copy(); od=ImageDraw.Draw(overview)
for k,g in enumerate(groups,1):
 i=g[0]; c=cs[i]; mask=masks[i]; cid='P%02d'%k; folder=out/cid; folder.mkdir()
 yy,xx=np.nonzero(mask)
 bounds=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)] if len(xx) else [int(v) for v in c['box_xyxy_original']]
 x1,y1,x2,y2=bounds; pad=max(64,int(max(x2-x1,y2-y1)*0.15))
 roi=[max(0,x1-pad),max(0,y1-pad),min(w,x2+pad),min(h,y2+pad)]
 assert roi[0]<roi[2] and roi[1]<roi[3]
 original=im.crop(roi); local=mask[roi[1]:roi[3],roi[0]:roi[2]]; arr=np.asarray(original).copy()
 arr[local]=(arr[local]*0.55+np.array([255,45,45])*0.45).astype(np.uint8)
 marked=Image.fromarray(arr); md=ImageDraw.Draw(marked); md.text((5,5),cid,fill='yellow',font=font,stroke_width=2,stroke_fill='black')
 original.save(folder/'original_context.jpg',quality=96); marked.save(folder/'marked_context.jpg',quality=96)
 Image.fromarray(mask.astype(np.uint8)*255).save(folder/'mask_original.png')
 Image.fromarray(local.astype(np.uint8)*255).save(folder/'mask_context.png')
 card=Image.new('RGB',(1024,570),'white'); d=ImageDraw.Draw(card)
 d.rectangle((0,0,1024,36),fill='black'); d.text((10,5),cid+' | original / predicted mask | raw '+','.join(cs[j]['id'] for j in g),fill='white',font=font)
 for panel,x in [(original,0),(marked,512)]:
  thumb=ImageOps.contain(panel,(500,500)); card.paste(thumb,(x+(512-thumb.width)//2,45+(500-thumb.height)//2))
 card.save(folder/'comparison.jpg',quality=94); cards.append(card)
 od.rectangle(bounds,outline='red',width=4); od.text((max(0,x1),max(0,y1-30)),cid,fill='yellow',font=font,stroke_width=2,stroke_fill='black')
 item={'id':cid,'representative_raw_id':c['id'],'equivalent_raw_ids':[cs[j]['id'] for j in g],'source_tile':c['tile_id'],'mask_bounds_original':bounds,'context_xyxy_original':roi,'context_origin_xy':roi[:2],'mask_area_pixels':areas[i],'detector_score':c['detector_score'],'sam_mask_score':c['sam_mask_score'],'original_context':cid+'/original_context.jpg','marked_context':cid+'/marked_context.jpg','mask':cid+'/mask_original.png','semantic_decision':'NOT_REVIEWED'}
 (folder/'candidate.json').write_text(json.dumps(item,indent=2)); records.append(item)
overview.save(out/'numbered_overview_full.jpg',quality=95); overview.thumbnail((1600,1000)); overview.save(out/'numbered_overview.jpg',quality=94)
contact=Image.new('RGB',(1024,285*max(1,math.ceil(len(cards)/2))),'#dddddd')
for i,card in enumerate(cards): contact.paste(card.resize((512,285)),((i%2)*512,(i//2)*285))
contact.save(out/'candidate_contact_sheet.jpg',quality=94)
with (out/'raw_mask_iou.csv').open('w',newline='') as f:
 writer=csv.writer(f); writer.writerow(['id']+[c['id'] for c in cs])
 for i,c in enumerate(cs): writer.writerow([c['id']]+ious[i].tolist())
report={'status':'CANDIDATES_PREPARED','stage':3,'source_result':str(src/'result.json'),'image':r['image'],'image_sha256':r['sha256'],'raw_count':n,'organized_count':len(groups),'near_duplicate_count':n-len(groups),'duplicate_rule':'mask IoU >= 0.85 for every group member; highest detector-score representative; engineering initial setting','semantic_rejections':0,'ground_truth_used':False,'vlm_run':False,'partial_overlaps_may_remain':True,'candidate_ids_are_not_physical_wire_ids':True,'all_raw_predictions_preserved':True,'candidates':records}
(out/'candidates.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='candidates'},indent=2),flush=True)
