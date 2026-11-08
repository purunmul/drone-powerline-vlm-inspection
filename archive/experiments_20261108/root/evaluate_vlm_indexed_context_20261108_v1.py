import csv
import json
from collections import Counter
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

from evaluate_first_image_v1 import metrics,readmask
from evaluate_sam_auto_points_20261108_v1 import display_lines
from run_vlm_indexed_context_20261108_v1 import BASE,STEMS,OUT,METHODS,BACKGROUND,input_paths,digest,read_json,save_json

POLICIES=('all_candidates',*METHODS,*(m+'_ungated' for m in METHODS))

def classify(record):
    value=record['parsed'] or {}
    if value.get('target_present') is not True:
        return 'recheck'
    decision=value.get('decision')
    if decision=='background' and value.get('category') in BACKGROUND:
        return 'background'
    if decision=='keep' and value.get('category')=='wire':
        return 'keep'
    return 'recheck'


def evaluate():
    manifest=read_json(OUT/'manifest.json')
    responses=[json.loads(l) for l in (OUT/'responses.jsonl').read_text().splitlines()]
    assert len(responses)==len(manifest)==read_json(OUT/'inference_complete.json')['records']==108
    assert len({r['image_tokens'] for r in responses})==1
    for a,b in zip(manifest,responses):
        assert all(a[k]==b[k] for k in a)
    result=OUT/'review'
    result.mkdir(exist_ok=False)
    rows=[]; decisions=[]; controls=[]; sample_inputs=set()
    examples={STEMS[0]:'P02',STEMS[1]:'P01',STEMS[2]:'P21'}
    for stem in STEMS:
        raw_dir,source=input_paths(stem)
        data=read_json(source/'candidates.json')['candidates']
        if stem==STEMS[0]:
            evaluation=BASE/'diagnostic_eval'/f'{stem}_v1'
        else:
            evaluation=BASE/'dev_transfer_20261107_v1'/stem/'evaluation'
        ref=readmask(evaluation/'reference_centerlines.png')
        refdist=distance_transform_edt(~ref)
        old=BASE/'skeleton_order_ablation_20261107_v1'/stem
        line_by_id={c['id']:readmask(old/'candidate_skeletons'/f"{c['id']}.png") for c in data}
        baseline=readmask(old/'A_all_candidates/skeleton_then_union/centerline.png')
        assert np.array_equal(baseline,np.logical_or.reduce(list(line_by_id.values())))
        by_key={(r['candidate'],r['method'],r['control']):r for r in responses if r['stem']==stem}
        assert len(by_key)==2*len(data)+2
        gate={}
        for method in METHODS:
            record=by_key[data[0]['id'],method,True]
            value=record['parsed'] or {}
            gate[method]=value.get('target_present') is False and value.get('decision')=='recheck'
            controls.append(dict(stem=stem,method=method,passed=gate[method],raw=record['raw']))
            sample_inputs.add(record['image'])
        sets={k:[] for k in POLICIES}
        sets['all_candidates']=list(line_by_id)
        for c in data:
            ident=c['id']
            sha=digest(source/c['mask'])
            line=line_by_id[ident]
            near_ref=int((refdist[line]<=5).sum())
            for method in METHODS:
                record=by_key[ident,method,False]
                assert record['mask_sha256']==sha
                action=classify(record)
                exclude=action=='background' and gate[method]
                if not exclude:
                    sets[method].append(ident)
                if action!='background':
                    sets[method+'_ungated'].append(ident)
                if ident==examples[stem] or action=='background':
                    sample_inputs.add(record['image'])
                decisions.append(dict(stem=stem,method=method,candidate=ident,
                    parsed=record['parsed'] is not None,proposed=action,excluded=exclude,
                    category=(record['parsed'] or {}).get('category'),reason=(record['parsed'] or {}).get('reason'),
                    centerline_pixels=int(line.sum()),centerline_near_reference_pixels_at5=near_ref,input=record['image']))
        folder=result/stem
        folder.mkdir()
        photo=Image.open(read_json(raw_dir/'result.json')['image']).convert('RGB')
        before=distance_transform_edt(~baseline)[ref]
        for policy,ids in sets.items():
            line=np.zeros_like(ref)
            for ident in ids:
                line|=line_by_id[ident]
            assert not (line&~baseline).any()
            scores,dist=metrics(line,ref,refdist)
            for score in scores:
                t=score['tolerance_pixels']
                after=dist[ref]<=t if dist is not None else np.zeros(ref.sum(),dtype=bool)
                assert not (after & (before>t)).any()
                score.update(stem=stem,policy=policy,kept=len(ids),removed=len(data)-len(ids),
                    lost_reference_pixels=int((~after&(before<=t)).sum()))
                rows.append(score)
            if policy in ('all_candidates','gray','indexed_context'):
                Image.fromarray(line.astype(np.uint8)*255).save(folder/f'{policy}_centerline.png')
            if policy=='indexed_context':
                display_lines(photo,line,(21,237,218)).save(folder/'prediction_on_original.jpg',quality=83)
        print('SCORED',stem[:8],{k:len(v) for k,v in sets.items()},gate,flush=True)
    fields=('centerline_precision','centerline_recall','centerline_f1')
    macro={}
    for t in (3,5,10):
        macro[t]={}
        for policy in POLICIES:
            subset=[r for r in rows if r['tolerance_pixels']==t and r['policy']==policy]
            macro[t][policy]={k:float(np.mean([r[k] for r in subset])) for k in fields}
            macro[t][policy].update({k:sum(r[k] for r in subset) for k in ('kept','removed','lost_reference_pixels')})
    for name,records in [('centerline_metrics',rows),('decisions',decisions)]:
        with (result/f'{name}.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=records[0]);writer.writeheader();writer.writerows(records)
    breakdown={m:{'proposals':dict(Counter(d['proposed'] for d in decisions if d['method']==m)),
        'valid':sum(d['parsed'] for d in decisions if d['method']==m),
        'excluded':sum(d['excluded'] for d in decisions if d['method']==m)} for m in METHODS}
    save_json(result/'summary.json',dict(rows=rows,macro=macro,controls=controls,decisions=decisions,breakdown=breakdown,
        same_number_of_vision_tokens=responses[0]['image_tokens'],candidate_weights_changed=False,
        ungated='diagnostic only, shows what would happen ignoring empty-target control',
        sample_inputs=sorted(sample_inputs)))
    with ZipFile('vlm_indexed_context_20261108_v1.zip','x',ZIP_DEFLATED) as z:
        for path in sorted(result.rglob('*')):
            if path.is_file():
                z.write(path,path.relative_to(result))
        for name in ('manifest.json','responses.jsonl','run_config.json','inference_complete.json'):
            z.write(OUT/name,name)
        selected=[p for p in sorted(sample_inputs) if '_empty.' in p or Path(p).name.startswith(examples[Path(p).parent.name]+'_')]
        for rel in selected:
            z.write(OUT/rel,'inputs/'+rel)
        for name in ('run_vlm_indexed_context_20261108_v1.py','evaluate_vlm_indexed_context_20261108_v1.py','batch_vlm_indexed_context_20261108_v1.py'):
            z.write(name,name)
        assert z.testzip() is None
    for policy,score in macro[5].items():
        print(policy,[round(score[k]*100,2) for k in fields],score['removed'],flush=True)
    print('EVALUATION_COMPLETE',len(rows),flush=True)

if __name__=='__main__':
    evaluate()
