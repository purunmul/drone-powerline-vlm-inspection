import json,time,hashlib,zipfile,importlib.metadata
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageOps
import torch
from lang_sam.models.sam import SAM
ROOT=Path('results/langsam_baseline/BEGO7553_frame_001098_candidates_v1')
OUT=Path('results/selective_reprocess/BEGO7553_frame_001098_round1')
PROMPTS=[
{'parent_id':'P01','crop':[1480,520,1660,1177],'box':[1525,550,1620,1150],'positive':[[1541,600],[1553,700],[1578,900],[1602,1100]],'negative':[[1500,800],[1630,900]]},
{'parent_id':'P05','crop':[1600,520,1800,1177],'box':[1635,550,1763,1150],'positive':[[1655,600],[1673,700],[1713,900],[1743,1100]],'negative':[[1620,850],[1630,1050]]},
{'parent_id':'P10','crop':[1860,1136,2080,1800],'box':[1890,1180,2040,1760],'positive':[[1919,1250],[1954,1400],[2020,1700]],'negative':[[1875,1300],[2065,1450]]},
{'parent_id':'P11','crop':[1880,1136,2120,1800],'box':[1925,1180,2065,1760],'positive':[[1948,1250],[1982,1400],[2044,1700]],'negative':[[1895,1350],[2100,1500]]}]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def overlay(im,mask,color):
    a=np.asarray(im).copy(); a[mask]=(a[mask]*0.48+np.array(color)*0.52).astype('uint8');return Image.fromarray(a)
def main():
    OUT.mkdir(parents=True,exist_ok=False)
    data=json.loads((ROOT/'candidates.json').read_text()); byid={c['id']:c for c in data['candidates']}
    image=Image.open(data['image']).convert('RGB'); assert sha(Path(data['image']))==data['image_sha256']
    before={c['id']:sha(ROOT/c['mask']) for c in data['candidates']}
    checkpoint=Path('.cache/torch/hub/checkpoints/sam2.1_hiera_small.pt');assert checkpoint.is_file()
    config={'stage':5,'round':1,'source_candidates':str(ROOT/'candidates.json'),'image':data['image'],'image_sha256':data['image_sha256'],'sam_type':'sam2.1_hiera_small','checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint),'prompt_source':'assistant selected points and boxes by inspecting original pixels; no XML or reference labels read','human_assisted':True,'partial_region_repair':True,'target':'one visible strand segment per parent region, not exhaustive wire recovery','selection':'highest SAM predicted mask score among three outputs; all alternatives retained','multimask_output':True,'prompts_original_xy':PROMPTS,'versions':{k:importlib.metadata.version(k) for k in ['torch','torchvision','numpy','Pillow']}}
    (OUT/'run_config.json').write_text(json.dumps(config,indent=2))
    sam=SAM(); t=time.perf_counter();sam.build_model('sam2.1_hiera_small',ckpt_path=str(checkpoint));load=time.perf_counter()-t
    print('SAM_LOADED',load,flush=True);torch.cuda.reset_peak_memory_stats();records=[];cards=[]
    for prompt in PROMPTS:
        pid=prompt['parent_id'];cid=pid+'_R1';p=OUT/cid;p.mkdir()
        x0,y0,x1,y1=prompt['crop'];crop=image.crop((x0,y0,x1,y1));w,h=crop.size
        pts=np.array(prompt['positive']+prompt['negative'],dtype='float32')-np.array([x0,y0],dtype='float32')
        labels=np.array([1]*len(prompt['positive'])+[0]*len(prompt['negative']),dtype='int32')
        box=np.array(prompt['box'],dtype='float32')-np.array([x0,y0,x0,y0],dtype='float32')
        assert np.all(pts[:,0]>=0) and np.all(pts[:,0]<w) and np.all(pts[:,1]>=0) and np.all(pts[:,1]<h)
        torch.cuda.synchronize();t=time.perf_counter()
        with torch.inference_mode():
            sam.predictor.set_image(np.asarray(crop))
            masks,scores,logits=sam.predictor.predict(point_coords=pts,point_labels=labels,box=box,multimask_output=True)
        torch.cuda.synchronize();elapsed=time.perf_counter()-t
        masks=np.asarray(masks,dtype=bool);scores=np.asarray(scores);assert masks.shape[1:]==(h,w)
        index=int(np.argmax(scores));new=masks[index]
        oldfull=np.asarray(Image.open(ROOT/byid[pid]['mask']).convert('L'))>0;old=oldfull[y0:y1,x0:x1]
        full=np.zeros((image.height,image.width),dtype=bool);full[y0:y1,x0:x1]=new;assert int(full.sum())==int(new.sum())
        crop.save(p/'original.jpg',quality=95);Image.fromarray(new.astype('uint8')*255).save(p/'mask_local.png');Image.fromarray(full.astype('uint8')*255).save(p/'mask_original.png')
        Image.fromarray(old.astype('uint8')*255).save(p/'before_mask_same_scope.png')
        np.savez_compressed(p/'all_sam_outputs.npz',masks=masks,scores=scores,logits=logits)
        promptview=crop.copy();draw=ImageDraw.Draw(promptview);draw.rectangle(tuple(box),outline='yellow',width=2)
        for (x,y),label in zip(pts,labels):draw.ellipse((x-4,y-4,x+4,y+4),fill='lime' if label else 'red')
        promptview.save(p/'prompts.jpg',quality=95)
        left=overlay(crop,old,(255,0,0));right=overlay(crop,new,(0,220,255))
        left.save(p/'before_overlay.jpg',quality=95);right.save(p/'after_overlay.jpg',quality=95)
        isolated=Image.composite(crop,Image.new('RGB',crop.size,(150,150,150)),Image.fromarray(new.astype('uint8')*255));isolated.save(p/'isolated_target.jpg',quality=95)
        panels=[crop,left,right];canvas=Image.new('RGB',(w*3,h+45),'white');dr=ImageDraw.Draw(canvas)
        for j,(im,title) in enumerate(zip(panels,['ORIGINAL','BEFORE (red)','AFTER SAM (cyan)'])):
            canvas.paste(im,(j*w,45));dr.text((j*w+5,8),cid+' '+title,fill='black')
        canvas.save(p/'comparison.jpg',quality=95);card=canvas.copy();card.thumbnail((900,740));cards.append(card)
        hits=[bool(new[int(y),int(x)]) for x,y in pts]
        r={'id':cid,'parent_id':pid,'round':1,'crop_xyxy_original':prompt['crop'],'context_origin_xy':[x0,y0],'prompt':prompt,'selected_output_index':index,'sam_scores':scores.tolist(),'before_area_same_scope':int(old.sum()),'after_area_same_scope':int(new.sum()),'area_change_ratio':float(new.sum()/old.sum()) if old.any() else None,'positive_prompt_hits':sum(hits[:len(prompt['positive'])]),'positive_prompt_count':len(prompt['positive']),'negative_prompt_hits':sum(hits[len(prompt['positive']):]),'inference_seconds':elapsed,'mask_original':str(p/'mask_original.png'),'mask_local':str(p/'mask_local.png'),'isolated_target':str(p/'isolated_target.jpg'),'comparison':str(p/'comparison.jpg'),'status':'SAM_INFERENCE_COMPLETE_NOT_ACCEPTED','finalized':False,'accuracy_evaluated':False,'parent_replaced':False,'unprocessed_parent_regions_retained':True}
        records.append(r);(p/'result.json').write_text(json.dumps(r,indent=2));print(cid,'AREA',r['before_area_same_scope'],r['after_area_same_scope'],'PROMPT_HITS',r['positive_prompt_hits'],'/',r['positive_prompt_count'],'SCORES',scores.tolist(),flush=True)
    assert before=={c['id']:sha(ROOT/c['mask']) for c in data['candidates']}
    sheet=Image.new('RGB',(max(c.width for c in cards),sum(c.height for c in cards)+40*len(cards)),'white');dr=ImageDraw.Draw(sheet);y=0
    for c,r in zip(cards,records):
        dr.text((5,y+5),r['id']+' | same crop before/after | automatic acceptance: pending',fill='black');sheet.paste(c,(0,y+35));y+=c.height+40
    sheet.save(OUT/'before_after_contact_sheet.jpg',quality=95)
    review_notes={'P01':'broad road/background mask','P02':'thin linear candidate; model pole classification needs context','P03':'visually wire-like strand; do not auto-reject as pole','P04':'thin strand with attached equipment; inspect mask extent','P05':'broad road/background mask','P06':'thin linear candidate; electrical type unconfirmed','P07':'horizontal narrow object; inspect scene context','P08':'thin linear candidate; electrical type unconfirmed','P09':'thin linear candidate; electrical type unconfirmed','P10':'broad strip adjacent to wires','P11':'broad strip adjacent to wires','P12':'possible building-edge/trim confusion','P13':'clearly visible vehicle'}
    (OUT/'visual_review_13.json').write_text(json.dumps({'source':'assistant visual inspection of original image and candidate overlays; no XML','notes':review_notes},indent=2))
    s={'status':'SELECTIVE_SAM_COMPLETE','stage':5,'round':1,'candidate_count':4,'all_13_source_masks_unchanged':True,'ground_truth_used':False,'human_assisted':True,'exhaustive_wire_recovery':False,'sam_load_seconds':load,'inference_seconds_total':sum(r['inference_seconds'] for r in records),'torch_peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'results':records}
    (OUT/'result.json').write_text(json.dumps(s,indent=2))
    with zipfile.ZipFile('selective_sam_round1.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in OUT.rglob('*'):
            if p.is_file():z.write(p,'round1/'+str(p.relative_to(OUT)))
        z.write('run_selective_sam_round1.py','run_selective_sam_round1.py')
    print('SELECTIVE_SAM_COMPLETE',len(records),'SECONDS',s['inference_seconds_total'],flush=True)
if __name__=='__main__':main()
