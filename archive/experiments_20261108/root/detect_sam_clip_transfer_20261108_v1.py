import csv
import os
import subprocess
from pathlib import Path

from sam_clip_transfer_config_20261108_v1 import ROOT,STEMS
from run_sam_auto_points_20261108_v1 import digest,read_json,save_json

sources=('run_langsam.sh','run_langsam_baseline.py','run_langsam_tiled.py','prepare_langsam_candidates.py',
         'run_sam_auto_points_20261108_v1.py','run_sam_narrow_strip_20261108_v1.py','evaluate_first_image_v1.py')
prior=read_json(Path('results/dev_transfer_20261107_v1/frozen_config.json'))
for name in sources:
    if name in prior['source_hashes']:
        assert digest(Path(name))==prior['source_hashes'][name], name
with Path('data/initial_8/selection.csv').open(newline='',encoding='utf-8-sig') as f:
    selected=list(csv.DictReader(f))
assert len(selected)==8 and tuple(Path(r['image']).stem for r in selected[3:])==STEMS
images={stem:Path('data/initial_8/images')/f'{stem}.jpg' for stem in STEMS}
ROOT.mkdir(exist_ok=False)
save_json(ROOT/'config.json',dict(images=STEMS,image_sha256={stem:digest(p) for stem,p in images.items()},
    source_sha256={name:digest(Path(name)) for name in sources},
    detection={k:prior[k] for k in ('prompt','box_threshold','text_threshold','tile_size','minimum_overlap','duplicate_iou')},
    clip_half_width=24,clip_end_margin=24,reference_only_in_evaluation=True,vlm_used=False))
prepare=Path('prepare_langsam_candidates.py').read_text()
env=os.environ.copy()
for key in ('PYTHONPATH','PYTHONHOME'):
    env.pop(key,None)
env.update(PYTHONNOUSERSITE='1',HF_HUB_OFFLINE='1')
for stem,image in images.items():
    folder=ROOT/stem
    folder.mkdir()
    whole,tiles,candidates=(folder/k for k in ('whole','tiles','candidates'))
    generated=folder/'prepare.py'
    adapted=prepare.replace("src=Path('results/langsam_baseline/BEGO7553_frame_001098_tiles1024_v1')",f'src=Path({str(tiles)!r})')
    adapted=adapted.replace("out=Path('results/langsam_baseline/BEGO7553_frame_001098_candidates_v1')",f'out=Path({str(candidates)!r})')
    assert str(tiles) in adapted and str(candidates) in adapted
    generated.write_text(adapted)
    commands=(('whole',['bash','run_langsam.sh','run_langsam_baseline.py','--image',str(image),
                         '--output',str(whole),'--prompt','power line.','--box-threshold','0.30','--text-threshold','0.25']),
              ('tiles',['bash','run_langsam.sh','run_langsam_tiled.py','--baseline',str(whole/'result.json'),
                         '--output',str(tiles),'--tile-size','1024','--overlap','256']),
              ('candidates',['.venv-langsam/bin/python',str(generated)]))
    for stage,cmd in commands:
        print('START',stem[:8],stage,flush=True)
        with (folder/f'{stage}.log').open('w') as f:
            subprocess.run(cmd,env=env,check=True,stdout=f,stderr=subprocess.STDOUT)
    data=read_json(candidates/'candidates.json')
    print('READY',stem[:8],data['raw_count'],data['organized_count'],flush=True)
save_json(ROOT/'detection_complete.json',dict(images=STEMS,status='COMPLETE'))
print('DETECTION_COMPLETE',flush=True)
