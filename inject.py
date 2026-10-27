#!/usr/bin/python3
import os,sys,hashlib,json,shutil
from pathlib import Path
D=Path(__file__).resolve().parent
args=sys.argv[1:];p=next((Path(x) for x in args if x.endswith('.pto')),None)
if p is None:os.execv(os.environ['PTOAS_ROOT']+'/build/tools/ptoas/ptoas',['ptoas']+args)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
known={sha(D/f'{i}.pto'):i for i in [0,10,20]}
assert sha(p) in known,(p,sha(p));i=known[sha(p)];src=D/f'{i}.layout.fatobj.o';dst=Path(args[args.index('-o')+1]);shutil.copy2(src,dst)
with (D/'injections.jsonl').open('a') as f:f.write(json.dumps({'case':i,'pto_sha':sha(p),'object_sha':sha(src),'output':str(dst)})+'\n')
