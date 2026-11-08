import json,hashlib,zipfile
from pathlib import Path
root=Path('results/vlm_review'); names={1:'BEGO7553_frame_001098_review_v1_final',2:'BEGO7553_frame_001098_review_v2',3:'BEGO7553_frame_001098_review_v3',4:'BEGO7553_frame_001098_review_v4'}
rows=[]
for v,name in names.items():
    p=root/name; d=json.loads((p/'review.json').read_text()); assert d['candidate_count']==13
    assert len({r['candidate_id'] for r in d['results']})==13
    assert sum(d['counts'].values())==13
    if v>=3:
        assert d['valid_json_count']==13 and d['source_masks_unchanged']
        for r in d['results']:
            assert r['image_tokens']>0 and r['finish_reason']=='stop'
            assert hashlib.sha256(Path(r['input']).read_bytes()).hexdigest()==r['input_sha256']
    rows.append({'version':v,'counts':d['counts'],'seconds':d.get('candidate_inference_seconds_total',d.get('inference_seconds_total')),'model_load_seconds':d['model_load_seconds'],'empty_control_pass':d.get('empty_control_pass')})
a={'status':'VERIFIED','versions':rows,'ground_truth_used':False,'accuracy_evaluated':False,'recommended_prototype':'v4 with manual review before finalization','notes':['v1/v2/v3/v4 are exploratory iterations on one development image','Multiple changes were made; no single-change causal effect or held-out performance claimed','v3 failed the empty-target control and is not adopted','v4 control pass is necessary but not sufficient for accuracy']}
Path('vlm_improvement_audit.json').write_text(json.dumps(a,indent=2))
with zipfile.ZipFile('vlm_improvement_20261106.zip','w',zipfile.ZIP_DEFLATED) as z:
    for v,name in names.items():
        for p in (root/name).iterdir():
            if p.is_file():z.write(p,'v'+str(v)+'/'+p.name)
        script=Path('review_langsam_candidates_v'+str(v)+'.py')
        if script.exists():z.write(script,'scripts/'+script.name)
    for name in ['vlm_review_v2.log','vlm_review_v3.log','vlm_review_v4.log','vlm_candidate_review_final.log']:
        p=Path('logs')/name;z.write(p,'logs/'+name)
    z.write('vlm_improvement_audit.json','vlm_improvement_audit.json')
print(json.dumps(a,indent=2));print('ZIP_BYTES',Path('vlm_improvement_20261106.zip').stat().st_size)
