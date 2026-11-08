import json,hashlib,shutil,zipfile
from pathlib import Path
p=Path('results/vlm_review/BEGO7553_frame_001098_review_v1_final')
d=json.loads((p/'review.json').read_text())
assert d['status']=='REVIEW_COMPLETE' and d['candidate_count']==13 and d['valid_json_count']==13
assert len({r['candidate_id'] for r in d['results']})==13
for r in d['results']:
    assert len(r['input_images'])==3 and r['finish_reason']=='stop' and r['parse_error'] is None
    for x in r['input_images']:
        assert hashlib.sha256(Path(x['path']).read_bytes()).hexdigest()==x['sha256']
    assert r['input_images'][1]['sha256'] != r['input_images'][2]['sha256']
src=Path('results/langsam_baseline/BEGO7553_frame_001098_candidates_v1')
shutil.copy2(src/'P13/comparison.jpg',p/'P13_false_keep_example.jpg')
refs=list(Path('.cache/huggingface/hub').glob('models--Qwen*/refs/main'))
a={'status':'OUTPUT_AND_INPUT_INTEGRITY_VERIFIED','count':13,'counts':d['counts'],'input_image_hashes_checked':39,'inference_seconds_total':d['inference_seconds_total'],'model_load_seconds_including_download':d['model_load_seconds'],'model_revisions':{str(x):x.read_text().strip() for x in refs},'ground_truth_used':False,'accuracy_evaluated':False,'human_visual_observations':[{'candidate_id':'P01','issue':'road-sized overbroad mask was kept'},{'candidate_id':'P05','issue':'road-sized overbroad mask was kept'},{'candidate_id':'P13','issue':'clearly visible car mask was kept'}],'conclusion':'Execution completed but the initial VLM reviewer failed to filter obvious non-wire or overbroad masks. Do not automatically finalize these masks.','command':'env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 HF_HOME=/home/work/123-0125/.cache/huggingface HF_HUB_DISABLE_XET=1 .venv-vlm/bin/python -u review_langsam_candidates_v1.py'}
(p/'review_audit.json').write_text(json.dumps(a,indent=2))
with zipfile.ZipFile('vlm_review_v1_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
    for x in p.iterdir():
        if x.is_file(): z.write(x,'review/'+x.name)
    z.write('review_langsam_candidates_v1.py','review_langsam_candidates_v1.py')
    z.write('logs/vlm_candidate_review_final.log','vlm_candidate_review_final.log')
print(json.dumps({k:a[k] for k in ['status','counts','input_image_hashes_checked','inference_seconds_total','model_load_seconds_including_download']},indent=2))
print('EXIT',Path('logs/vlm_candidate_review_final.exit').read_text().strip())
print('BUNDLE',Path('vlm_review_v1_bundle.zip').stat().st_size)
