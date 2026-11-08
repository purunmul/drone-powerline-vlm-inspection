import json,csv,hashlib,time,zipfile,shutil
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np,cv2
from PIL import Image
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from evaluate_first_image_v1 import metrics,trace,readmask,sanity_tests
ROOT=Path('results/dev_transfer_20261107_v1')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 sanity_tests();assert (ROOT/'phase2_complete.json').exists()
 frozen=json.loads((ROOT/'frozen_config.json').read_text());assert frozen['source_hashes']=={n:sha(Path(n)) for n in frozen['source_hashes']}
 tree=ET.parse('data/initial_8/annotations.xml');combined=[];summaries=[]
 expected={'PUYL5573_frame_000165':23,'KZIV5445_frame_000488':21}
 for name in frozen['images']:
  folder=ROOT/name;src=folder/'candidates';out=folder/'evaluation';out.mkdir(exist_ok=False)
  data=json.loads((src/'candidates.json').read_text());image=Image.open(data['image']).convert('RGB');w,h=image.size;assert (w,h)==(3840,2160);assert sha(Path(data['image']))==frozen['image_hashes'][name]
  candidates=data['candidates'];masks={c['id']:readmask(src/c['mask']) for c in candidates};hashes={c['id']:sha(src/c['mask']) for c in candidates}
  raw=json.loads((folder/'tiles/result.json').read_text());assert raw['status']=='INFERENCE_COMPLETE'
  review=json.loads((folder/'vlm/review.json').read_text());policy=json.loads((folder/'vlm/conservative_policy.json').read_text());assert set(masks)=={r['candidate_id'] for r in review['results']}=={r['candidate_id'] for r in policy['results']}
  anns=[a for a in tree.getroot().findall('image') if Path(a.attrib['name']).stem==name];assert len(anns)==1;ann=anns[0];assert [int(ann.attrib[k]) for k in ['width','height']]==[w,h]
  gt=np.zeros((h,w),np.uint8);vectors=[];clipped=0
  for i,line in enumerate(ann.findall('polyline')):
   assert line.attrib['label'] in {'powerline','polyline'};pts=np.array([[float(x) for x in pair.split(',')] for pair in line.attrib['points'].split(';')],dtype=np.float64);assert len(pts)>=2
   raster=np.rint(pts).astype(np.int32);clipped+=int(((raster[:,0]<0)|(raster[:,0]>=w)|(raster[:,1]<0)|(raster[:,1]>=h)).sum());raster[:,0]=np.clip(raster[:,0],0,w-1);raster[:,1]=np.clip(raster[:,1],0,h-1);cv2.polylines(gt,[raster],False,1,1,lineType=cv2.LINE_8)
   vectors.append({'annotation_fragment_id':i+1,'label':line.attrib['label'],'original_points_xy':pts.tolist()})
  assert len(vectors)==expected[name];gt=gt.astype(bool);gd=distance_transform_edt(~gt)
  Image.fromarray(gt.astype('uint8')*255).save(out/'reference_centerlines.png');(out/'reference_vectors.json').write_text(json.dumps(vectors,indent=2));only=ET.Element('annotations');only.append(ET.fromstring(ET.tostring(ann)));ET.ElementTree(only).write(out/'reference.xml',encoding='utf-8',xml_declaration=True)
  def union(ids):
   m=np.zeros((h,w),bool)
   for ident in ids:m|=masks[ident]
   return m
  allids=list(masks);kept=[r['candidate_id'] for r in review['results'] if r['decision']=='keep'];not_rejected=[r['candidate_id'] for r in policy['results'] if r['decision']!='reject']
  organized=union(allids);rawunion=readmask(folder/'tiles/union_mask.png');whole=np.zeros((h,w),bool)
  with np.load(folder/'whole/prediction.npz') as z:
   for m in z['masks']:whole|=m.astype(bool)
  conditions=[('A_all_candidates',organized,allids),('B_vlm_keep_only',union(kept),kept),('C_remove_vehicle_only',union(not_rejected),not_rejected),('Z_raw_tile_union',rawunion,None),('W_whole_image',whole,None)]
  bg=(np.asarray(image)*.7).astype('uint8');kernel=np.ones((5,5),np.uint8);condition_summaries=[]
  for label,mask,ids in conditions:
   dest=out/label;dest.mkdir();sk=skeletonize(mask);ms,pd=metrics(sk,gt,gd);paths=trace(sk)
   for r in ms:r.update(image=name,condition=label,mask_pixels=int(mask.sum()))
   combined.extend(ms);Image.fromarray(mask.astype('uint8')*255).save(dest/'mask.png');Image.fromarray(sk.astype('uint8')*255).save(dest/'centerline.png')
   assert all(0<=x<w and 0<=y<h for p in paths for x,y in p['points_xy'])
   (dest/'line_coordinates.json').write_text(json.dumps({'image_size_wh':[w,h],'coordinate_origin_xy':[0,0],'coordinate_unit':'original image pixel','not_physical_wire_instance_ids':True,'graph_path_count':len(paths),'paths':paths},separators=(',',':')))
   show=bg.copy();near=sk&(gd<=5);far=sk&(gd>5);miss=gt&(pd>5) if pd is not None else gt
   for m,color in [(miss,(255,205,0)),(far,(235,30,190)),(near,(0,220,255))]:show[cv2.dilate(m.astype('uint8'),kernel)>0]=color
   view=Image.fromarray(show);view.thumbnail((1600,900));view.save(dest/'reference_comparison.jpg',quality=95)
   condition_summaries.append({'condition':label,'candidate_ids':ids,'graph_path_count':len(paths),'metrics':ms})
   mid=ms[1];print(name,label,'P/R/F1',mid['centerline_precision'],mid['centerline_recall'],mid['centerline_f1'],flush=True)
  diag=[]
  for ident,mask in masks.items():
   sk=skeletonize(mask);n=int(sk.sum());diag.append({'id':ident,'mask_pixels':int(mask.sum()),'skeleton_pixels':n,'skeleton_support_at5':float((gd[sk]<=5).mean()) if n else 0.})
  assert hashes=={c['id']:sha(src/c['mask']) for c in candidates}
  summary={'image':name,'status':'DEVELOPMENT_TRANSFER_EVALUATED','reference_fragments':len(vectors),'reference_union_pixels':int(gt.sum()),'border_points_clipped':clipped,'raw_candidate_count':data['raw_count'],'organized_candidate_count':data['organized_count'],'near_duplicate_count':data['near_duplicate_count'],'organized_union_equals_raw':bool(np.array_equal(organized,rawunion)),'raw_union_pixels_removed_by_duplicate_filter':int((rawunion&~organized).sum()),'whole_image_candidates':json.loads((folder/'whole/result.json').read_text())['candidate_count'],'vlm_counts':review['counts'],'conservative_counts':policy['counts'],'empty_control_pass':review['empty_control_pass'],'source_masks_unchanged':True,'conditions':condition_summaries,'candidate_diagnostics':diag}
  (out/'summary.json').write_text(json.dumps(summary,indent=2));summaries.append(summary)
 # Compare only matched automatic conditions with first image, excluding human assistance and manual visual selection.
 first=json.loads(Path('results/diagnostic_eval/BEGO7553_frame_001098_v1/summary.json').read_text());matched=['A_all_candidates','B_vlm_keep_only','C_remove_vehicle_only'];comparison=[]
 for s in first['conditions']:
  if s['condition'] in matched:
   for r in s['metrics']:comparison.append(dict(r,image='BEGO7553_frame_001098'))
 comparison.extend(r for r in combined if r['condition'] in matched)
 keys=['image','condition','tolerance_pixels','centerline_precision','centerline_recall','centerline_f1','pred_centerline_pixels','reference_union_pixels','matched_pred_pixels','covered_reference_pixels','mask_pixels']
 for fname,rows in [('metrics_all_conditions.csv',combined),('three_image_comparison.csv',comparison)]:
  with (ROOT/fname).open('w',newline='') as f:
   writer=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
 aggregate=[]
 for label in matched:
  for tol in [3,5,10]:
   rows=[r for r in comparison if r['condition']==label and r['tolerance_pixels']==tol];assert len(rows)==3
   denom_p=sum(r['pred_centerline_pixels'] for r in rows);denom_r=sum(r['reference_union_pixels'] for r in rows);p=sum(r['matched_pred_pixels'] for r in rows)/denom_p if denom_p else 0.;r=sum(r['covered_reference_pixels'] for r in rows)/denom_r;f=2*p*r/(p+r) if p+r else 0.
   aggregate.append({'condition':label,'tolerance_pixels':tol,'macro_precision':sum(x['centerline_precision'] for x in rows)/3,'macro_recall':sum(x['centerline_recall'] for x in rows)/3,'macro_f1':sum(x['centerline_f1'] for x in rows)/3,'pooled_precision':p,'pooled_recall':r,'pooled_f1':f})
 result={'status':'TWO_ADDITIONAL_DEVELOPMENT_IMAGES_COMPLETE','new_images':summaries,'three_image_aggregate':aggregate,'all_source_settings_unchanged':True,'heldout_160_used':False,'human_assisted_repair_run':False,'limitations':['three development images; first repeatedly tuned','no autonomous selective repair implemented or credited','centerline pixel agreement at tolerance, not physical wire instance accuracy','duplicate-filter effects reported separately','5px summary is a diagnostic convention; all 3/5/10 tolerances saved','no training, connection, damage, or electrical correctness assessment']}
 (ROOT/'summary.json').write_text(json.dumps(result,indent=2));shutil.copy2('results/diagnostic_eval/BEGO7553_frame_001098_v1/summary.json',ROOT/'first_image_previous_summary.json')
 for name in ['run_dev_transfer_phase1.py','run_dev_transfer_phase2.py','run_dev_transfer_phase3.py']:shutil.copy2(name,ROOT/'frozen_sources'/name)
 assert frozen['source_hashes']=={n:sha(Path(n)) for n in frozen['source_hashes']}
 with zipfile.ZipFile('dev_transfer_20261107_v1.zip','w',zipfile.ZIP_DEFLATED) as z:
  for p in ROOT.rglob('*'):
   if p.is_file():z.write(p,'dev_transfer/'+str(p.relative_to(ROOT)))
 print('TRANSFER_COMPLETE SANITY_TESTS_OK COORDINATES_COVER_SKELETON SOURCE_FILES_UNCHANGED HELDOUT_NOT_USED',flush=True)
if __name__=='__main__':main()
