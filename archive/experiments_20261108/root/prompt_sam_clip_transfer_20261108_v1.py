from sam_clip_transfer_config_20261108_v1 import ROOT,STEMS
import replay_sam_auto_points_20261108_v1 as runner

assert runner.read_json(ROOT/'detection_complete.json')['images']==list(STEMS)
runner.STEMS=STEMS
runner.OUT=ROOT/'full_tile_prompts'
runner.input_paths=lambda stem: (ROOT/stem/'tiles',ROOT/stem/'candidates')
runner.infer()
