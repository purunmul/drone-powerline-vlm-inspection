import os,json,time
from pathlib import Path
root=Path('/home/work/123-0125')
os.environ['HF_HOME']=str(root/'.cache/huggingface')
os.environ['TORCH_HOME']=str(root/'.cache/torch')
import torch
from lang_sam import LangSAM
assert torch.cuda.is_available(), 'CUDA unavailable'
torch.cuda.reset_peak_memory_stats()
started=time.perf_counter()
model=LangSAM(sam_type='sam2.1_hiera_small')
torch.cuda.synchronize()
load_seconds=time.perf_counter()-started
try:
 from sam2 import _C
 extension='available'
except (ImportError,OSError) as exc:
 extension='unavailable: '+str(exc)
report={'status':'MODEL_LOAD_OK','scope':'model loading only; no detection accuracy measured','sam_type':model.sam_type,'grounding_dino':'IDEA-Research/grounding-dino-base','torch':torch.__version__,'cuda_runtime':torch.version.cuda,'sam_device':str(next(model.sam.model.parameters()).device),'gdino_device':str(next(model.gdino.model.parameters()).device),'download_and_load_seconds':load_seconds,'gpu_allocated_gib':torch.cuda.memory_allocated()/2**30,'gpu_peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'sam_cuda_extension':extension}
(root/'logs/langsam_model_check.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2),flush=True)
