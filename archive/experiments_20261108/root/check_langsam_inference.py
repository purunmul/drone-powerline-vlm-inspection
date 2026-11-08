import os,json,time
from pathlib import Path
root=Path('/home/work/123-0125')
os.environ['HF_HOME']=str(root/'.cache/huggingface')
os.environ['TORCH_HOME']=str(root/'.cache/torch')
import torch,numpy as np
from PIL import Image
from lang_sam import LangSAM
model=LangSAM(sam_type='sam2.1_hiera_small')
image=Image.open(root/'test_image.jpg').convert('RGB')
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
started=time.perf_counter()
result=model.predict([image],['cat.'])[0]
torch.cuda.synchronize()
seconds=time.perf_counter()-started
count=len(result['boxes'])
masks=np.asarray(result['masks'])
if count: assert len(masks)==count and tuple(masks.shape[-2:])==(image.height,image.width)
report={'status':'INFERENCE_OK' if count else 'DETECTION_EMPTY','scope':'existing generic cat test image only; no powerline accuracy measured','image':'test_image.jpg','image_size':list(image.size),'prompt':'cat.','candidate_count':count,'mask_shape':list(masks.shape),'inference_seconds':seconds,'gpu_peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'sam_cuda_extension':('available' if __import__('importlib.util',fromlist=['find_spec']).find_spec('sam2._C') else 'unavailable; optional postprocessing may be skipped')}
np.savez_compressed(root/'logs/langsam_generic_smoke.npz',boxes=result['boxes'],masks=masks,scores=result['scores'])
(root/'logs/langsam_generic_smoke.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2),flush=True)
