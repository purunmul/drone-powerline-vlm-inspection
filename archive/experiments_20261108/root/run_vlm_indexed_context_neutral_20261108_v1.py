import json
from pathlib import Path

from run_vlm_indexed_context_20261108_v1 import OUT as INPUTS, STEMS, CATEGORIES, BACKGROUND

OUT=INPUTS.with_name('vlm_indexed_context_neutral_20261108_v1')
PROMPT=(
    'Classify the selected pixels in the upper-left TARGET panel. In an isolated panel the non-gray pixels are selected. '
    'In a photo panel a visible cyan outline marks the selection. The text header is only an ID. '
    'If no selection is visible in the upper-left panel, answer no_selection, even if there are visible objects elsewhere. '
    'Use any other panels only for context. Classify the selected object itself. If it contains both a wire and a background object, use mixed. '
    'Return one JSON object with target_id (copy {target}), category, reason (up to 15 words). '
    'Category choices: no_selection, vehicle, road_or_ground, pole_hardware, building, vegetation, other_background, wire, mixed, unclear.'
)

def parse(output,target,finish):
    try:
        data=json.JSONDecoder().raw_decode(output[output.index('{'):])[0]
        category=data['category']
        assert finish=='stop' and data['target_id']==target and category in CATEGORIES|{'no_selection'}
        assert isinstance(data['reason'],str) and data['reason'].strip()
        visible=category!='no_selection'
        decision='background' if category in BACKGROUND else 'keep' if category=='wire' else 'recheck'
        return dict(data,target_present=visible,decision=decision)
    except (AssertionError,ValueError,KeyError,TypeError):
        return None


def main():
    import batch_vlm_indexed_context_20261108_v1 as runner
    OUT.mkdir(exist_ok=False)
    for stem in STEMS:
        (OUT/stem).symlink_to((INPUTS/stem).resolve(),target_is_directory=True)
    manifest=INPUTS/'manifest.json'
    (OUT/manifest.name).write_bytes(manifest.read_bytes())
    assert len(list(OUT.glob('*/*.jpg')))==108
    runner.OUT=OUT
    runner.PROMPT=PROMPT
    runner.parse=parse
    runner.main()

if __name__=='__main__':
    main()
