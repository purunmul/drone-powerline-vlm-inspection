import json,csv,hashlib,zipfile,shutil,time,importlib.metadata
from pathlib import Path
import numpy as np,cv2
from PIL import Image,ImageDraw,ImageFont
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from evaluate_first_image_v1 import metrics,trace,readmask,sanity_tests
OUT=Path('results/skeleton_order_ablation_20261107_v1')
TR=Path('results/dev_transfer_20261107_v1')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 sanity_tests();OUT.mkdir(parents=True,exist_ok=False)
 datasets=[('BEGO7553_frame_001098',Path('results/langsam_baseline/BEGO7553_frame_001098_candidates_v1'),Path('results/vlm_review/BEGO7553_frame_001098_review_v4/review.json'),Path('results/diagnostic_eval/BEGO7553_frame_001098_v1'))]
 for name in ['PUYL5573_frame_000165','KZIV5445_frame_000488']:datasets.append((name,TR/name/'candidates',TR/name/'vlm/review.json',TR/name/'evaluation'))
 sourcefiles=[Path('evaluate_first_image_v1.py')]
 for _,src,rev,ev in datasets:
  sourcefiles.extend([src/'candidates.json',rev,ev/'summary.json',ev/'reference_centerlines.png'])
  for c in json.loads((src/'candidates.json').read_text())['candidates']:sourcefiles.append(src/c['mask'])
 sources={str(p):sha(p) for p in sourcefiles}
 config={'images':[x[0] for x in datasets],'conditions':['A_all_candidates','B_vlm_keep_only'],'methods':{'union_then_skeleton':'skeletonize(logical OR of selected masks); reproduces previous output','skeleton_then_union':'logical OR of individually skeletonized candidate masks; no second skeletonization, pruning, smoothing, line fitting or GT-based selection'},'skeleton_method':'skimage skeletonize default 2-D Zhang','tolerances_pixels':[3,5,10],'candidate_selection':'same frozen v4 strict keep list as previous diagnostics','source_hashes':sources,'versions':{k:importlib.metadata.version(k) for k in ['numpy','scipy','scikit-image','opencv-python-headless','Pillow']},'no_new_model_inference':True,'heldout_160_used':False,'coordinate_caution':'8-neighbor paths are graph segments, not physical wire identities; union of per-candidate skeletons may retain nearby parallel duplicate traces and multi-pixel junctions','development_only':True}
 (OUT/'run_config.json').write_text(json.dumps(config,indent=2));allrows=[];summaries=[];font=ImageFont.load_default(size=22)
 for name,src,rev,ev in datasets:
  start=time.perf_counter();dest=OUT/name;dest.mkdir();(dest/'candidate_skeletons').mkdir()
  data=json.loads((src/'candidates.json').read_text());review=json.loads(rev.read_text());ids=[c['id'] for c in data['candidates']];kept=[r['candidate_id'] for r in review['results'] if r['decision']=='keep'];assert set(kept)<=set(ids)
  masks={c['id']:readmask(src/c['mask']) for c in data['candidates']};sks={}
  for ident in ids:
   sks[ident]=skeletonize(masks[ident]);assert not (sks[ident]&~masks[ident]).any();Image.fromarray(sks[ident].astype('uint8')*255).save(dest/'candidate_skeletons'/(ident+'.png'))
  def union(items,selected):
   m=np.zeros_like(next(iter(items.values())))
   for ident in selected:m|=items[ident]
   return m
  outputs={};selected_masks={}
  for label,selected in [('A_all_candidates',ids),('B_vlm_keep_only',kept)]:
   m=union(masks,selected);old=skeletonize(m);new=union(sks,selected);assert np.array_equal(m,readmask(ev/label/'mask.png'));assert np.array_equal(old,readmask(ev/label/'centerline.png'));assert not (new&~m).any()
   outputs[(label,'union_then_skeleton')]=old;outputs[(label,'skeleton_then_union')]=new;selected_masks[label]=m
  assert not (outputs[('B_vlm_keep_only','skeleton_then_union')]&~outputs[('A_all_candidates','skeleton_then_union')]).any()
  # Reference is loaded only after both prediction variants have been generated.
  gt=readmask(ev/'reference_centerlines.png');gd=distance_transform_edt(~gt);image=Image.open(data['image']).convert('RGB');w,h=image.size;assert (w,h)==(3840,2160);assert sha(Path(data['image']))==data['image_sha256'];bg=(np.asarray(image)*.7).astype('uint8');kernel=np.ones((5,5),np.uint8)
  previous=json.loads((ev/'summary.json').read_text());prev={s['condition']:s['metrics'] for s in previous['conditions']};msummaries=[];coverages={}
  for (label,method),sk in outputs.items():
   folder=dest/label/method;folder.mkdir(parents=True);ms,pd=metrics(sk,gt,gd);paths=trace(sk);assert all(0<=x<w and 0<=y<h for p in paths for x,y in p['points_xy'])
   for r in ms:r.update(image=name,condition=label,method=method)
   if method=='union_then_skeleton':
    for r in ms:
     old=next(x for x in prev[label] if x['tolerance_pixels']==r['tolerance_pixels'])
     for k in ['centerline_precision','centerline_recall','centerline_f1']:assert abs(r[k]-old[k])<1e-12
   allrows.extend(ms);coverages[(label,method)]={t:(pd[gt]<=t) if pd is not None else np.zeros(int(gt.sum()),bool) for t in [3,5,10]}
   Image.fromarray(sk.astype('uint8')*255).save(folder/'centerline.png');(folder/'line_coordinates.json').write_text(json.dumps({'image_size_wh':[w,h],'coordinate_origin_xy':[0,0],'method':method,'not_physical_wire_instance_ids':True,'candidate_ids':ids if label=='A_all_candidates' else kept,'graph_path_count':len(paths),'paths':paths},separators=(',',':')))
   near=sk&(gd<=5);far=sk&(gd>5);miss=gt&(pd>5) if pd is not None else gt;arr=bg.copy()
   for mask,color in [(miss,(255,205,0)),(far,(235,30,190)),(near,(0,220,255))]:arr[cv2.dilate(mask.astype('uint8'),kernel)>0]=color
   view=Image.fromarray(arr);view.thumbnail((1600,900));view.save(folder/'reference_comparison.jpg',quality=94)
   mid=ms[1];msummaries.append({'condition':label,'method':method,'graph_path_count':len(paths),'metrics':ms});print(name,label,method,'P/R/F1',mid['centerline_precision'],mid['centerline_recall'],mid['centerline_f1'],flush=True)
  exclusion=[]
  for method in ['union_then_skeleton','skeleton_then_union']:
   for tol in [3,5,10]:
    ac=coverages[('A_all_candidates',method)][tol];bc=coverages[('B_vlm_keep_only',method)][tol];lost=int((ac&~bc).sum());gained=int((~ac&bc).sum())
    if method=='skeleton_then_union':assert gained==0
    exclusion.append({'method':method,'tolerance_pixels':tol,'reference_pixels_lost_after_keep_only':lost,'reference_pixels_gained_after_keep_only':gained})
  for label in ['A_all_candidates','B_vlm_keep_only']:
   board=Image.new('RGB',(1600,500),'black');draw=ImageDraw.Draw(board)
   for i,method in enumerate(['union_then_skeleton','skeleton_then_union']):
    pic=Image.open(dest/label/method/'reference_comparison.jpg');pic.thumbnail((800,450));board.paste(pic,(800*i,50));draw.text((800*i+10,10),method,font=font,fill='white')
   board.save(dest/(label+'_method_comparison.jpg'),quality=94)
  summary={'image':name,'candidate_count':len(ids),'keep_count':len(kept),'frozen_keep_ids':kept,'previous_results_reproduced_exactly':True,'new_keep_centerline_is_subset_of_new_all_centerline':True,'exclusion_comparison':exclusion,'conditions':msummaries,'processing_seconds_including_io':time.perf_counter()-start};summaries.append(summary);(dest/'summary.json').write_text(json.dumps(summary,indent=2))
  shutil.copy2(src/'candidates.json',dest/'source_candidates.json');shutil.copy2(rev,dest/'source_vlm_review.json')
 aggregates=[]
 for label in ['A_all_candidates','B_vlm_keep_only']:
  for method in ['union_then_skeleton','skeleton_then_union']:
   for tol in [3,5,10]:
    rows=[r for r in allrows if r['condition']==label and r['method']==method and r['tolerance_pixels']==tol];assert len(rows)==3
    n=sum(r['pred_centerline_pixels'] for r in rows);g=sum(r['reference_union_pixels'] for r in rows);p=sum(r['matched_pred_pixels'] for r in rows)/n if n else 0.;r=sum(r['covered_reference_pixels'] for r in rows)/g;f=2*p*r/(p+r) if p+r else 0.
    aggregates.append({'condition':label,'method':method,'tolerance_pixels':tol,'macro_precision':sum(r['centerline_precision'] for r in rows)/3,'macro_recall':sum(r['centerline_recall'] for r in rows)/3,'macro_f1':sum(r['centerline_f1'] for r in rows)/3,'pooled_precision':p,'pooled_recall':r,'pooled_f1':f})
 with (OUT/'metrics.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(allrows[0]));writer.writeheader();writer.writerows(allrows)
 result={'status':'SKELETON_ORDER_ABLATION_COMPLETE','images':summaries,'aggregates':aggregates,'old_results_reproduced_exactly':True,'source_masks_and_vlm_decisions_unchanged':True,'no_new_model_inference':True,'heldout_160_used':False,'limitations':['three development images from potentially related capture locations','per-candidate skeleton union may retain nearby duplicate traces; no physical-wire identity evaluation','pixel-distance agreement is not instance segmentation, connectivity or electrical correctness','no topology repair or reference-informed pruning','method choice remains a development decision requiring later frozen evaluation']}
 (OUT/'summary.json').write_text(json.dumps(result,indent=2));shutil.copy2('compare_skeleton_order_v1.py',OUT/'compare_skeleton_order_v1.py');shutil.copy2('evaluate_first_image_v1.py',OUT/'metric_helpers_original.py');assert sources=={p:sha(Path(p)) for p in sources}
 with zipfile.ZipFile('skeleton_order_ablation_20261107_v1.zip','w',zipfile.ZIP_DEFLATED) as z:
  for p in OUT.rglob('*'):
   if p.is_file():z.write(p,'skeleton_order_ablation/'+str(p.relative_to(OUT)))
 print('ABLATION_COMPLETE OLD_RESULTS_REPRODUCED MASKS_DECISIONS_UNCHANGED COORDINATE_COVERAGE_OK SUBSET_INVARIANT_OK NO_MODEL_RUNS',flush=True)
if __name__=='__main__':main()
