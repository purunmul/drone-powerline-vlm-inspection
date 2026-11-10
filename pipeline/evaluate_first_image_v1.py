from repository_paths import image_path
import json,csv,time,hashlib,zipfile,subprocess,importlib.metadata
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np,cv2
from PIL import Image,ImageDraw
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
ROOT=Path('results/langsam_baseline/BEGO7553_frame_001098_candidates_v1')
REPAIR=Path('results/selective_reprocess/BEGO7553_frame_001098_round1')
OUT=Path('results/diagnostic_eval/BEGO7553_frame_001098_v1')
TOLS=[3,5,10]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def readmask(p):return np.asarray(Image.open(p).convert('L'))>0
def metrics(sk,gt,refdist):
    n=int(sk.sum()); g=int(gt.sum());dist=distance_transform_edt(~sk) if n else None
    out=[]
    for t in TOLS:
        hitp=int((refdist[sk]<=t).sum()) if n else 0;hitr=int((dist[gt]<=t).sum()) if n and g else 0
        p=hitp/n if n else 0.;r=hitr/g if g else 0.;f=2*p*r/(p+r) if p+r else 0.
        out.append({'tolerance_pixels':t,'centerline_precision':p,'centerline_recall':r,'centerline_f1':f,'pred_centerline_pixels':n,'reference_union_pixels':g,'matched_pred_pixels':hitp,'covered_reference_pixels':hitr})
    return out,dist
def trace(sk):
    h,w=sk.shape; nodes=set(map(int,np.flatnonzero(sk)));adj={}
    for u in nodes:
        y,x=divmod(u,w);adj[u]=[yy*w+xx for yy in range(max(0,y-1),min(h,y+2)) for xx in range(max(0,x-1),min(w,x+2)) if (yy!=y or xx!=x) and yy*w+xx in nodes]
    seen=set();paths=[]
    def edge(a,b):return (min(a,b),max(a,b))
    def walk(a,b):
        path=[a,b];seen.add(edge(a,b));prev,cur=a,b
        while len(adj[cur])==2:
            nxt=adj[cur][0] if adj[cur][0]!=prev else adj[cur][1]
            if edge(cur,nxt) in seen:break
            path.append(nxt);seen.add(edge(cur,nxt));prev,cur=cur,nxt
        return path
    for a in sorted(nodes):
        if len(adj[a])==0:paths.append([a])
        elif len(adj[a])!=2:
            for b in adj[a]:
                if edge(a,b) not in seen:paths.append(walk(a,b))
    for a in sorted(nodes):
        for b in adj[a]:
            if edge(a,b) not in seen:paths.append(walk(a,b))
    assert {u for p in paths for u in p}==nodes
    return [{'segment_id':i+1,'closed':len(p)>2 and p[0]==p[-1],'points_xy':[[u%w,u//w] for u in p]} for i,p in enumerate(paths)]
def sanity_tests():
    gt=np.zeros((30,40),bool);gt[10,5:30]=True;refdist=distance_transform_edt(~gt)
    exact,_=metrics(gt,gt,refdist);assert all(x['centerline_f1']==1 for x in exact)
    shifted=np.zeros_like(gt);shifted[16,5:30]=True;m,_=metrics(shifted,gt,refdist)
    assert m[0]['centerline_f1']==0 and m[1]['centerline_f1']==0 and m[2]['centerline_f1']==1
    empty,_=metrics(np.zeros_like(gt),gt,refdist);assert all(x['centerline_f1']==0 for x in empty)
    assert trace(gt)[0]['points_xy']
def main():
    sanity_tests();OUT.mkdir(parents=True,exist_ok=False)
    d=json.loads((ROOT/'candidates.json').read_text());rows=d['candidates'];impath=image_path(d['image']);assert sha(impath)==d['image_sha256']
    image=Image.open(impath).convert('RGB');w,h=image.size;assert (w,h)==(3840,2160)
    masks={c['id']:readmask(ROOT/c['mask']) for c in rows};sources={str(ROOT/c['mask']):sha(ROOT/c['mask']) for c in rows}
    xml=Path('data/initial_8/annotations.xml');tree=ET.parse(xml);matches=[x for x in tree.getroot().findall('image') if Path(x.attrib['name']).name==impath.name];assert len(matches)==1
    ann=matches[0];assert int(ann.attrib['width'])==w and int(ann.attrib['height'])==h
    gt=np.zeros((h,w),np.uint8);vectors=[];clipped=0
    for j,line in enumerate(ann.findall('polyline')):
        assert line.attrib['label'] in {'powerline','polyline'}
        pts=np.array([[float(v) for v in pair.split(',')] for pair in line.attrib['points'].split(';')],dtype=np.float64)
        assert len(pts)>=2
        raster=np.rint(pts).astype(np.int32);clipped+=int(((raster[:,0]<0)|(raster[:,0]>=w)|(raster[:,1]<0)|(raster[:,1]>=h)).sum());raster[:,0]=np.clip(raster[:,0],0,w-1);raster[:,1]=np.clip(raster[:,1],0,h-1)
        cv2.polylines(gt,[raster],False,1,1,lineType=cv2.LINE_8);vectors.append({'annotation_fragment_id':j+1,'label':line.attrib['label'],'original_points_xy':pts.tolist()})
    assert len(vectors)==27;gt=gt.astype(bool);refdist=distance_transform_edt(~gt)
    Image.fromarray(gt.astype('uint8')*255).save(OUT/'reference_centerlines.png');(OUT/'reference_vectors.json').write_text(json.dumps(vectors,indent=2))
    only=ET.Element('annotations');only.append(ET.fromstring(ET.tostring(ann)));ET.ElementTree(only).write(OUT/'reference_first_image.xml',encoding='utf-8',xml_declaration=True)
    def union(ids):
        a=np.zeros((h,w),bool)
        for ident in ids:a|=masks[ident]
        return a
    baseline=union(list(masks));stored=readmask(Path('results/langsam_baseline/BEGO7553_frame_001098_tiles1024_v1/union_mask.png'));assert np.array_equal(baseline,stored)
    v4=json.loads(Path('results/vlm_review/BEGO7553_frame_001098_review_v4/review.json').read_text());kept=[r['candidate_id'] for r in v4['results'] if r['decision']=='keep']
    policy=json.loads(Path('results/vlm_review/BEGO7553_frame_001098_review_v4_1/review.json').read_text());not_rejected=[r['candidate_id'] for r in policy['results'] if r['decision']!='reject']
    reviewed_kept=['P02','P03','P09']
    fixed={k:v.copy() for k,v in masks.items()};rep=json.loads((REPAIR/'result.json').read_text())
    for r in rep['results']:
        x0,y0,x1,y1=r['crop_xyxy_original'];local=readmask(Path(r['mask_local']));assert local.shape==(y1-y0,x1-x0);fixed[r['parent_id']][y0:y1,x0:x1]=local
    manual=np.zeros((h,w),bool)
    for a in fixed.values():manual|=a
    conditions=[('A_all_candidates',list(masks),baseline,'All 13 original candidate masks'),('B_vlm_keep_only',kept,union(kept),'Diagnostic strict acceptance: uncertain candidates omitted; not an approved deployment'),('C_remove_vehicle_only',not_rejected,union(not_rejected),'Conservative policy: retain uncertain originals; remove only rejected vehicle'),('D_visual_segments_only',reviewed_kept,union(reviewed_kept),'Non-independent assistant visual selection frozen in prior review; not automatic'),('E_hypothetical_manual_partial_repair',list(masks),manual,'Hypothetical human-assisted replacement inside four parent crops only; four repairs were not approved; outside crops retained')]
    config={'date':'2026-11-07','image':str(impath),'image_sha256':sha(impath),'annotation_source':str(xml),'annotation_source_sha256':sha(xml),'reference_fragments':len(vectors),'reference_raster_union_pixels':int(gt.sum()),'border_points_clipped_for_rasterization':clipped,'tolerances_pixels':TOLS,'display_tolerance_pixels':5,'skeleton_method':'skimage.morphology.skeletonize default 2-D Zhang; no pruning or smoothing','coordinate_output':'8-neighbor skeleton graph paths in original integer XY; graph branches are not physical wire instances','metric_definition':'skeleton-pixel precision: prediction skeleton pixels <=t from reference union; recall: reference union pixels <=t from prediction skeleton; F1 harmonic mean; union pixel counts, not object counts','mask_band_metric':'fraction of foreground pixels inside reference distance band; auxiliary agreement proxy, not true segmentation precision because reference is polylines','selection_from_reference':False,'conditions':[{'name':n,'candidate_ids':ids,'interpretation':note} for n,ids,m,note in conditions],'development_only':True,'heldout_160_not_used':True,'no_new_model_inference':True,'versions':{k:importlib.metadata.version(k) for k in ['numpy','scipy','scikit-image','opencv-python-headless','Pillow']}}
    (OUT/'run_config.json').write_text(json.dumps(config,indent=2));(OUT/'packages.lock.txt').write_text(subprocess.check_output(['.venv-eval/bin/python','-m','pip','freeze'],text=True))
    allmetrics=[];summaries=[]
    bg=np.asarray(image).copy();bg=(bg*.7).astype('uint8');kernel=np.ones((5,5),np.uint8)
    for name,ids,mask,note in conditions:
        p=OUT/name;p.mkdir();t=time.perf_counter();sk=skeletonize(mask);result,pdist=metrics(sk,gt,refdist);paths=trace(sk)
        for r in result:
            tpx=r['tolerance_pixels'];r.update(condition=name,mask_pixels=int(mask.sum()),mask_fraction_within_reference_band=float((refdist[mask]<=tpx).mean()) if mask.any() else 0.,reference_coverage_by_mask=float((distance_transform_edt(~mask)[gt]<=tpx).mean()) if mask.any() else 0.)
        allmetrics.extend(result);Image.fromarray(mask.astype('uint8')*255).save(p/'mask.png');Image.fromarray(sk.astype('uint8')*255).save(p/'centerline.png')
        (p/'line_coordinates.json').write_text(json.dumps({'image_size_wh':[w,h],'coordinate_origin_xy':[0,0],'coordinate_unit':'original image pixel','not_physical_wire_instance_ids':True,'graph_path_count':len(paths),'paths':paths},separators=(',',':')))
        near=sk&(refdist<=5);far=sk&(refdist>5);miss=gt&(pdist>5) if pdist is not None else gt
        show=bg.copy()
        for m,color in [(miss,(255,205,0)),(far,(235,30,190)),(near,(0,220,255))]:show[cv2.dilate(m.astype('uint8'),kernel)>0]=color
        view=Image.fromarray(show);view.thumbnail((1600,900));view.save(p/'reference_comparison.jpg',quality=95)
        summary={'condition':name,'candidate_ids':ids,'interpretation':note,'metrics':result,'graph_path_count':len(paths),'seconds':time.perf_counter()-t};summaries.append(summary)
        mid=next(x for x in result if x['tolerance_pixels']==5);print(name,'P/R/F1',mid['centerline_precision'],mid['centerline_recall'],mid['centerline_f1'],'MASK',int(mask.sum()),flush=True)
    candidate_rows=[]
    for ident,mask in masks.items():
        sk=skeletonize(mask);n=int(sk.sum());candidate_rows.append({'id':ident,'mask_pixels':int(mask.sum()),'skeleton_pixels':n,'skeleton_support_at5':float((refdist[sk]<=5).mean()) if n else 0.,'mask_fraction_within_reference_band5':float((refdist[mask]<=5).mean()) if mask.any() else 0.})
    local_rows=[]
    for r in rep['results']:
        x0,y0,x1,y1=r['crop_xyxy_original'];g=gt[y0:y1,x0:x1];gd=distance_transform_edt(~g);old=masks[r['parent_id']][y0:y1,x0:x1];new=readmask(Path(r['mask_local']))
        for variant,mask in [('old',old),('repair',new)]:
            ms,_=metrics(skeletonize(mask),g,gd);local_rows.append({'id':r['id'],'variant':variant,'scope_xyxy':[x0,y0,x1,y1],'reference_scope':'all annotated line fragments inside this crop, not only the single prompted strand','mask_pixels':int(mask.sum()),'metrics':ms})
    assert sources=={p:sha(Path(p)) for p in sources}
    summary={'status':'DIAGNOSTIC_EVALUATION_COMPLETE','reference_fragments':len(vectors),'reference_union_pixels':int(gt.sum()),'conditions':summaries,'candidate_diagnostics':candidate_rows,'matched_crop_comparisons':local_rows,'source_masks_unchanged':True,'accuracy_is_development_diagnostic_not_heldout':True,'limitations':['single development image repeatedly inspected during tuning','annotation centerlines are not full segmentation masks','rasterization and tolerance affect scores','skeleton topology is provisional, no wire identity or electrical connectivity recovered','manual four-crop repairs target one strand and can reduce coverage of other reference lines','hypothetical repair condition is not an accepted production result','reference disagreement may include annotation omission or registration error']}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    with (OUT/'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(allmetrics[0]));writer.writeheader();writer.writerows(allmetrics)
    with (OUT/'candidate_diagnostics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(candidate_rows[0]));writer.writeheader();writer.writerows(candidate_rows)
    lines=['# First-image development diagnostic','', 'Tolerance=5 pixels for this summary; 3/5/10 all saved. Cyan=predicted skeleton within 5px of reference; magenta=predicted skeleton outside reference band; yellow=reference not recovered within 5px. These are annotation-agreement indicators, not proof of real-world wire identity.','', '|Condition|Precision|Recall|F1|Mask pixels|Mask fraction in reference band|','|---|---|---|---|---|---|']
    for s in summaries:
        x=next(r for r in s['metrics'] if r['tolerance_pixels']==5);lines.append('|'+s['condition']+'|'+ '|'.join(f'{x[k]:.6f}' for k in ['centerline_precision','centerline_recall','centerline_f1'])+'|'+str(x['mask_pixels'])+'|'+f'{x["mask_fraction_within_reference_band"]:.6f}'+'|')
    (OUT/'summary.md').write_text(chr(10).join(lines))
    with zipfile.ZipFile('first_image_diagnostic_20261107.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in OUT.rglob('*'):
            if p.is_file():z.write(p,'diagnostic/'+str(p.relative_to(OUT)))
        z.write('evaluate_first_image_v1.py','evaluate_first_image_v1.py')
    print('EVALUATION_COMPLETE','REFERENCE',len(vectors),'SANITY_TESTS_OK','GRAPH_COORDINATE_COVERAGE_OK','SOURCE_MASKS_UNCHANGED',flush=True)
if __name__=='__main__':main()
