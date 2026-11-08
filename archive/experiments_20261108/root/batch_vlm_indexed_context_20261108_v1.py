import importlib.metadata
import json
import time
import base64

from vllm import LLM,SamplingParams
from run_vlm_indexed_context_20261108_v1 import OUT,FIRST,METHODS,PROMPT,digest,parse,read_json,save_json


def message(entry):
    path=OUT/entry['image']
    assert digest(path)==entry['image_sha256']
    image='data:image/jpeg;base64,'+base64.b64encode(path.read_bytes()).decode('ascii')
    return [{'role':'user','content':[{'type':'image_url','image_url':{'url':image}},
             {'type':'text','text':PROMPT.format(target=entry['candidate'])}]}]


def main():
    baseline=read_json(FIRST/'run_config.json')
    settings=dict(baseline['settings'],gpu_memory_utilization=.27,max_model_len=2048,
                  max_num_batched_tokens=2048,max_num_seqs=4,quantization='bitsandbytes',load_format='bitsandbytes')
    sample=dict(temperature=0,max_tokens=160)
    config=dict(settings=settings,sampling=sample,prompt=PROMPT,methods=METHODS,
        control_rule='per photo and method, target_present=false and decision=recheck required for automatic exclusions',
        selection_rule='exclude only valid background decision with present target and explicit background category, under passing control',
        reference_used=False,training=False,source_checkpoint_unchanged=True,
        quantization='vLLM bitsandbytes on load, 4-bit',versions={k:importlib.metadata.version(k) for k in ('vllm','bitsandbytes','torch')})
    entries=read_json(OUT/'manifest.json')
    start=time.perf_counter()
    model=LLM(**settings)
    options=SamplingParams(**sample)
    token=model.get_tokenizer().convert_tokens_to_ids('<|image_pad|>')
    print('QWEN_READY',round(time.perf_counter()-start,2),flush=True)
    save_json(OUT/'run_config.json',config)
    with (OUT/'responses.jsonl').open('x') as log:
        for first in range(0,len(entries),4):
            batch=entries[first:first+4]
            responses=model.chat([message(e) for e in batch],sampling_params=options,use_tqdm=False)
            for entry,result in zip(batch,responses,strict=True):
                value=result.outputs[0]
                parsed=parse(value.text,entry['candidate'],value.finish_reason)
                tokens=result.prompt_token_ids.count(token)
                assert tokens>0
                record=dict(entry,raw=value.text,parsed=parsed,finish_reason=value.finish_reason,image_tokens=tokens)
                log.write(json.dumps(record,ensure_ascii=False)+'\n')
            log.flush()
            print('BATCH',first+len(batch),flush=True)
    save_json(OUT/'inference_complete.json',dict(records=len(entries),seconds=time.perf_counter()-start))
    print('INFERENCE_COMPLETE',len(entries),flush=True)

if __name__=='__main__':
    main()
