import os,json,hashlib,subprocess,importlib.util
from pathlib import Path
import torch,tilelang,ptodsl,ptoas
from tilelang.ascend import language as T
from tile_kernels import config as cfg
from tile_kernels.quant import norm_backward_asc as am
D=Path(__file__).resolve().parent
REPO=Path(os.environ['TK_BRANCH_ROOT'])
if os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD')=='0':
 cfg.get_device_num_sms=lambda:36
 import tile_kernels.testing.generator as gen
 gen.get_device_num_sms=lambda:36
spec=importlib.util.spec_from_file_location('repro_tests',REPO/'tests/quant/test_norm_backward.py');tests=importlib.util.module_from_spec(spec);spec.loader.exec_module(tests)
CASES=tests.generate_test_params(0)
assert [(CASES[i]['hidden'],CASES[i]['num_tokens'],CASES[i]['accumulate']) for i in [0,10,20]]==[(4096,8001,False),(5120,8001,False),(7168,8001,False)]
spec=importlib.util.spec_from_file_location('repro_vmi',D/'norm_backward_vmi.py');vm=importlib.util.module_from_spec(spec);spec.loader.exec_module(vm)
FACT={'asc':am.get_norm_backward_kernel_asc,'vmi':vm.get_norm_backward_kernel_vmi}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify():
 assert '--enable-vecscope-mem-bar=false' in os.environ.get('PTO_FLAGS','')
 assert subprocess.check_output(['git','-C',os.environ['PTOAS_ROOT'],'rev-parse','HEAD'],text=True).strip()=='105e48b046c9ca30fda8c978e1a00f030df0633d'
 for n in ['norm_backward_asc.py','norm_forward_vmi.py']:assert sha(D/n)==sha(REPO/'tile_kernels/quant'/n),n
 assert str(Path(ptodsl.__file__).resolve()).startswith(os.environ['PTOAS_ROOT'])
 for line in (D/'SHA256SUMS').read_text().splitlines():
  h,n=line.split('  ',1);assert sha(D/n)==h,n
 return True
def kwargs(p):return dict(num_tokens=p['num_tokens'],hidden=p['hidden'],with_weight=p['use_weight'],with_residual_out_grad=p['has_residual_out_grad'],weight_grad_initialized=p['accumulate'],out_scale=p['out_scale'],x_dtype=T.bfloat16,weight_grad_dtype=T.bfloat16,num_vec_cores=72)
def reset_seed(seed):
 from tile_kernels.rand.seed import set_seed
 set_seed(seed);torch.npu.manual_seed_all(seed)
