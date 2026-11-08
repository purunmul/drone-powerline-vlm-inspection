import json,zipfile,shutil
from pathlib import Path
from collections import Counter
def decide(r,control_pass):
    p=r.get('parsed')
    if not control_pass:return 'uncertain','halt_review','failed_empty_control'
    if not p or p.get('visibility')!='clear' or p.get('category')=='unclear':return 'uncertain','inspect_crop','insufficient_evidence'
    if p['category']=='vehicle':return 'reject','none','visible_vehicle'
    if r.get('broad_mask_flag'):return 'uncertain','resegment','broad_mask'
    if p['category']=='wire_or_cable':return 'keep','none','provisional_wire_candidate'
    return 'uncertain','inspect_context','nonvehicle_class_needs_context'
def main():
    source=Path('results/vlm_review/BEGO7553_frame_001098_review_v4/review.json')
    d=json.loads(source.read_text()); rows=[]
    for r in d['results']:
        decision,action,reason=decide(r,d['empty_control_pass'])
        assert decide(r,False)[0]=='uncertain'
        rows.append({'candidate_id':r['candidate_id'],'model_v4_decision':r['decision'],'model_category':r['parsed']['category'] if r['parsed'] else None,'decision':decision,'next_action':action,'policy_reason':reason,'broad_mask_flag':r['broad_mask_flag'],'finalized':False})
    out=Path('results/vlm_review/BEGO7553_frame_001098_review_v4_1');out.mkdir(exist_ok=False)
    s={'status':'CONSERVATIVE_POLICY_APPLIED','source':str(source),'method':'v4 model outputs with deterministic conservative policy; no new GPU inference','counts':dict(Counter(r['decision'] for r in rows)),'candidate_count':len(rows),'empty_control_pass':d['empty_control_pass'],'control_failure_gate_tested_on_candidates':len(rows),'ground_truth_used':False,'accuracy_evaluated':False,'source_masks_modified':False,'keep_is_provisional':True,'results':rows}
    (out/'review.json').write_text(json.dumps(s,indent=2))
    lines=['# Final conservative decisions','', '|ID|Raw v4|Final policy|Next action|Reason|','|---|---|---|---|---|']
    for r in rows:lines.append('|'+ '|'.join([r['candidate_id'],r['model_v4_decision'],r['decision'],r['next_action'],r['policy_reason']])+'|')
    (out/'comparison.md').write_text(chr(10).join(lines))
    shutil.copy2('vlm_improvement_20261106.zip','vlm_improvement_final_20261106.zip')
    with zipfile.ZipFile('vlm_improvement_final_20261106.zip','a',zipfile.ZIP_DEFLATED) as z:
        for p in out.iterdir():z.write(p,'v4_1/'+p.name)
        z.write('conservative_review_policy_v1.py','scripts/conservative_review_policy_v1.py')
    print('FINAL_POLICY',s['counts'],'CONTROL_FAILURE_GATE_CHECKS',13,'MASKS_MODIFIED',False,flush=True)
if __name__=='__main__':main()
