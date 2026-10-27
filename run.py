import sys,os,json,gc,statistics,hashlib
from pathlib import Path
import support as s
D=Path(__file__).resolve().parent;s.verify();basepath=os.environ['PATH'];loaded={}
def kernel(i,arm):
 if (i,arm) in loaded:return loaded[i,arm]
 cache=D/arm/'cache';tmp=D/arm/'tmp';cache.mkdir(parents=True,exist_ok=True);tmp.mkdir(parents=True,exist_ok=True)
 os.environ.update(TILELANG_CACHE_DIR=str(cache),TMPDIR=str(tmp),PATH=(str(D/'bin')+':'+basepath if arm=='layout' else basepath))
 import tilelang.cache as cm
 for c in list(cm._dispatch_map.values())+list(cm._ascend_dispatch_map.values()):c._memory_cache.clear()
 f=s.FACT['asc' if arm=='asc' else 'vmi']; f._kernel_cache.clear(); f._call_form_cache.clear(); k=f(**s.kwargs(s.CASES[i]));lib=getattr(k.adapter,'libpath',None); lib=Path(lib) if lib else next(cache.rglob('executable.so'),None)
 print('LOADED',i,arm,str(lib),s.sha(lib) if lib else 'fresh_compile',flush=True);loaded[i,arm]=k;return k
phase=sys.argv[1]
if phase=='compile':
 for i in [0,10,20]:
  for arm in ['asc','baseline','layout']:kernel(i,arm)
 print('COMPILE_OK',flush=True);sys.exit()
s.torch.npu.set_device(1)
from tilelang.profiler import do_bench
from tile_kernels.torch import norm_backward_ref
records=[]
for i in [0,10,20]:
 p=s.CASES[i];ks={a:kernel(i,a) for a in ['asc','baseline','layout']};rec={'id':i,'params':p,'checks':[]}
 for seed in [20261027+i,20261023,20261024]:
  s.reset_seed(seed);x,w,rstd,dy,res,dw=s.tests.generate_test_data(p);ref=norm_backward_ref(dy,x,w,rstd,res,p['out_scale']);calls={};outs={}
  for a,k in ks.items():
   dx=s.torch.empty_like(x);partial=s.torch.empty((72,p['hidden']),dtype=s.torch.float32,device=x.device);outdw=s.torch.empty_like(w)
   args=(x,w,dy,rstd,dx,partial,res,outdw);calls[a]=lambda k=k,args=args:k(*args);calls[a]();s.torch.npu.synchronize();outs[a]=(dx,outdw)
  check={'seed':seed}
  for a in ks:
   s.torch.testing.assert_close(outs[a][0],ref[0]);s.torch.testing.assert_close(outs[a][1],ref[1].to(outs[a][1].dtype));check[a+'_reference']='pass'
   check[a+'_vs_asc_elements']=[int((x!=y).sum()) for x,y in zip(outs[a],outs['asc'])]
  rec['checks'].append(check);print('CHECK',i,check,flush=True)
 times=[]
 for rnd in range(5):
  order=['asc','baseline','layout'];order=order[rnd%3:]+order[:rnd%3];row={}
  for a in order:row[a]=do_bench(calls[a],backend='msprof',warmup=0,rep=20)*1000
  times.append(row);print('TIME',i,rnd,row,flush=True)
 rec['times']=times;rec['median']={a:statistics.median(t[a] for t in times) for a in ks};records.append(rec);(D/'results.json').write_text(json.dumps(records,indent=2));print('RESULT',rec['median'],flush=True)
 del calls,outs;gc.collect();s.torch.npu.empty_cache()
print('DONE',flush=True)
