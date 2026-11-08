import json,csv,hashlib,zipfile,shutil,time
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np,cv2
from PIL import Image,ImageDraw,ImageOps,ImageFont
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from evaluate_first_image_v1 import readmask,sanity_tests
OUT=Path('results/failure_analysis_20261107_v1')
TRANSFER=Path('results/dev_transfer_20261107_v1')
TOLS=[3,5,10]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def coverage(mask,gt):
 if mask.any():return distance_transform_edt(~mask)[gt]
 return np.full(int(gt.sum()),np.inf)
def main():
 sanity_tests();OUT.mkdir(parents=True,exist_ok=False)
 datasets=[('BEGO7553_frame_001098',Path('results/langsam_baseline/BEGO7553_frame_001098_candidates_v1'),Path('results/vlm_review/BEGO7553_frame_001098_review_v4/review.json'),Path('results/vlm_review/BEGO7553_frame_001098_review_v4_1/review.json'),Path('results/diagnostic_eval/BEGO7553_frame_001098_v1'))]
 for name in ['PUYL5573_frame_000165','KZIV5445_frame_000488']:datasets.append((name,TRANSFER/name/'candidates',TRANSFER/name/'vlm/review.json',TRANSFER/name/'vlm/conservative_policy.json',TRANSFER/name/'evaluation'))
 config={'scope':'posthoc error diagnosis on three development images','tolerances_pixels':TOLS,'ranking':'descending count of reference pixels covered by an individual excluded-candidate skeleton and lost from the strict-keep union; ties by candidate ID','reference_used_for_diagnosis_and_case_selection':True,'not_a_deployable_selection_rule':True,'new_model_inference':False,'training':False,'source_masks_modified':False,'heldout_160_used':False,'metric_caution':'individual candidate skeleton support is annotation agreement, not binary candidate accuracy; supports overlap and category contributions must not be added; union skeletonization is nonlinear','source_hashes':{str(p):sha(p) for _,src,rev,pol,ev in datasets for p in [src/'candidates.json',rev,pol,ev/'reference_centerlines.png']}}
 (OUT/'run_config.json').write_text(json.dumps(config,indent=2));allrows=[];summaries=[]
 font=ImageFont.load_default(size=20)
 for name,src,rev,pol,ev in datasets:
  dest=OUT/name;dest.mkdir();data=json.loads((src/'candidates.json').read_text());reviews=json.loads(rev.read_text());policies=json.loads(pol.read_text());rs={r['candidate_id']:r for r in reviews['results']};ps={r['candidate_id']:r for r in policies['results']};gt=readmask(ev/'reference_centerlines.png');g=int(gt.sum());gd=distance_transform_edt(~gt)
  a=readmask(ev/'A_all_candidates/centerline.png');b=readmask(ev/'B_vlm_keep_only/centerline.png');amask=readmask(ev/'A_all_candidates/mask.png');ad=coverage(a,gt);bd=coverage(b,gt);md=coverage(amask,gt)
  losts={t:(ad<=t)&(bd>t) for t in TOLS};keepcover=bd<=5;lost=losts[5]
  records=[];dists={};hashes={c['id']:sha(src/c['mask']) for c in data['candidates']}
  for c in data['candidates']:
   ident=c['id'];mask=readmask(src/c['mask']);sk=skeletonize(mask);cd=coverage(sk,gt);dists[ident]=cd;rr=rs[ident];p=rr.get('parsed') or {};n=int(sk.sum())
   for tol in TOLS:
    hit=(cd<=tol)&losts[tol];r={'image':name,'id':ident,'raw_decision':rr['decision'],'conservative_decision':ps[ident]['decision'],'category':p.get('category','invalid'),'visibility':p.get('visibility','invalid'),'evidence':p.get('evidence',''),'broad_mask_flag':rr['broad_mask_flag'],'thickness_proxy':rr['thickness_proxy_p95_original_pixels'],'mask_pixels':int(mask.sum()),'skeleton_pixels':n,'tolerance_pixels':tol,'candidate_skeleton_precision':float((gd[sk]<=tol).mean()) if n else 0.,'reference_coverage_fraction':float((cd<=tol).mean()),'lost_reference_pixels_supported':int(hit.sum()) if rr['decision']!='keep' else 0,'lost_reference_fraction_supported':float(hit.sum()/losts[tol].sum()) if rr['decision']!='keep' and losts[tol].any() else 0.,'lost_reference_pp_supported':float(hit.sum()/g*100) if rr['decision']!='keep' else 0.}
    records.append(r)
   print('CANDIDATE_ANALYZED',name,ident,rr['decision'],flush=True)
  nonkeep=[c['id'] for c in data['candidates'] if rs[c['id']]['decision']!='keep'];summed=np.zeros(g,np.int16)
  for ident in nonkeep:summed+=(dists[ident]<=5).astype(np.int16)
  for r in records:
   if r['tolerance_pixels']==5:r['uniquely_supported_lost_reference_pixels']=int(((dists[r['id']]<=5)&lost&(summed==1)).sum()) if r['id'] in nonkeep else 0
   else:r['uniquely_supported_lost_reference_pixels']=None
  ranked=sorted([r for r in records if r['tolerance_pixels']==5 and r['raw_decision']!='keep'],key=lambda r:(-r['lost_reference_pixels_supported'],r['id']))
  category=[]
  for cat in sorted({rs[i].get('parsed',{}).get('category','invalid') if rs[i].get('parsed') else 'invalid' for i in nonkeep}):
   ids=[i for i in nonkeep if (rs[i].get('parsed') or {}).get('category','invalid')==cat];cov=np.zeros(g,bool)
   for ident in ids:cov|=dists[ident]<=5
   category.append({'category':cat,'candidate_ids':ids,'lost_reference_pixels_supported_by_any':int((cov&lost).sum()),'lost_reference_fraction_supported':float((cov&lost).sum()/lost.sum()) if lost.any() else 0.,'overlaps_other_categories':True})
  decomposition=[]
  for t in TOLS:
   ac=ad<=t;bc=bd<=t;mc=md<=t
   pieces={'covered_by_both':int((ac&bc).sum()),'lost_after_strict_keep':int((ac&~bc).sum()),'gained_after_strict_keep_due_to_reskeletonization':int((~ac&bc).sum()),'covered_by_neither':int((~ac&~bc).sum())};assert sum(pieces.values())==g
   decomposition.append({'tolerance_pixels':t,'reference_union_pixels':g,**pieces,'baseline_recall':float(ac.mean()),'strict_keep_recall':float(bc.mean()),'outside_baseline_mask_distance_band':int((~mc).sum()),'mask_band_overlap_but_skeleton_miss':int((mc&~ac).sum()),'mask_band_note':'foreground overlap proxy only, does not establish correct wire segmentation'})
  cases=[];image=Image.open(data['image']).convert('RGB');cmap={c['id']:c for c in data['candidates']}
  # Explicitly GT-informed inspection examples, not new inference prompts.
  for r in ranked[:3]:
   ident=r['id'];c=cmap[ident];roi=c['context_xyxy_original'];x0,y0,x1,y1=roi;original=image.crop(roi);mask=readmask(src/c['mask']);sk=skeletonize(mask);localmask=mask[y0:y1,x0:x1];localsk=sk[y0:y1,x0:x1];localgt=gt[y0:y1,x0:x1]
   isolated=Image.composite(original,Image.new('RGB',original.size,(150,150,150)),Image.fromarray(localmask.astype('uint8')*255))
   arr=(np.asarray(original)*.65).astype('uint8');near=localsk&(gd[y0:y1,x0:x1]<=5);far=localsk&~near;kernel=np.ones((3,3),np.uint8)
   for m,color in [(localgt,(255,205,0)),(far,(235,30,190)),(near,(0,220,255))]:arr[cv2.dilate(m.astype('uint8'),kernel)>0]=color
   board=Image.new('RGB',(1440,560),'#eeeeee');draw=ImageDraw.Draw(board);draw.rectangle((0,0,1440,64),fill='black');draw.text((10,5),name+' / '+ident+' / '+r['raw_decision']+' / '+r['category'],font=font,fill='white');draw.text((10,31),'candidate alignment='+format(r['candidate_skeleton_precision'],'.3f')+' | supported lost reference pixels='+str(r['lost_reference_pixels_supported'])+' (overlap allowed)',font=font,fill='white')
   for index,(panel,label) in enumerate([(original,'Original crop'),(isolated,'Selected foreground (VLM input view)'),(Image.fromarray(arr),'Reference + candidate skeleton')]):
    thumb=ImageOps.contain(panel,(470,450));board.paste(thumb,(index*480+(480-thumb.width)//2,98+(450-thumb.height)//2));draw.text((index*480+6,72),label,font=font,fill='black')
   path=dest/(ident+'_diagnostic.jpg');board.save(path,quality=94);cases.append({'id':ident,'diagnostic_board':str(path.relative_to(OUT)),'selection_uses_reference':True,'candidate':r})
  assert hashes=={c['id']:sha(src/c['mask']) for c in data['candidates']}
  summary={'image':name,'candidate_count':len(data['candidates']),'raw_counts':dict(Counter(r['decision'] for r in rs.values())),'conservative_counts':dict(Counter(r['decision'] for r in ps.values())),'decomposition':decomposition,'lost_pixels_with_support_from_any_individual_nonkeep_skeleton':int((lost&(summed>0)).sum()),'lost_pixels_without_individual_support_at5':int((lost&(summed==0)).sum()),'category_support_at5_overlaps_allowed':category,'ranked_nonkeep_candidates_at5':ranked,'inspection_cases':cases,'sources_unchanged':True}
  (dest/'summary.json').write_text(json.dumps(summary,indent=2));summaries.append(summary);allrows.extend(records)
  print('IMAGE_DONE',name,'LOST',int(lost.sum()),'TOP',[(r['id'],r['category'],round(r['candidate_skeleton_precision'],3),r['lost_reference_pixels_supported']) for r in ranked[:4]],flush=True)
 with (OUT/'candidate_diagnostics.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(allrows[0]));writer.writeheader();writer.writerows(allrows)
 result={'status':'THREE_IMAGE_FAILURE_DIAGNOSIS_COMPLETE','candidate_count':sum(s['candidate_count'] for s in summaries),'nonkeep_count':sum(sum(v for k,v in s['raw_counts'].items() if k!='keep') for s in summaries),'images':summaries,'new_model_inference':False,'ground_truth_informed_diagnosis':True,'heldout_160_used':False,'source_masks_modified':False}
 (OUT/'summary.json').write_text(json.dumps(result,indent=2));shutil.copy2('analyze_vlm_failure_v1.py',OUT/'analyze_vlm_failure_v1.py');assert config['source_hashes']=={p:sha(Path(p)) for p in config['source_hashes']}
 with zipfile.ZipFile('failure_analysis_20261107_v1.zip','w',zipfile.ZIP_DEFLATED) as z:
  for p in OUT.rglob('*'):
   if p.is_file():z.write(p,'failure_analysis/'+str(p.relative_to(OUT)))
 print('FAILURE_ANALYSIS_COMPLETE NO_MODEL_INFERENCE SOURCES_UNCHANGED',flush=True)
if __name__=='__main__':main()
