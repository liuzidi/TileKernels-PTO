from pathlib import Path
import re
D=Path(__file__).parent
for case,h in [(0,4096),(10,5120),(20,7168)]:
 s=(D/f'{case}.vpto').read_text();mask=re.findall(r'(%\w+) = pto.pset_b16 "PAT_ALL"',s)[-1];maps={};out=[];n=0
 for line in s.splitlines():
  m=re.match(r'(\s*)(%\w+), (%\w+) = pto.vintlv (%\w+), (%\w+) : !pto.vreg<128xbf16>',line)
  if m:
   i,lo,hi,z,src=m.groups();maps[lo]=(src,'EVEN');maps[hi]=(src,'ODD');n+=1;continue
  m=re.match(r'(\s*)(%\w+) = pto.vbitcast (%\w+) : !pto.vreg<128xbf16> -> !pto.vreg<64xf32>',line)
  if m and m[3] in maps:
   i,dst,key=m.groups();src,part=maps.pop(key);line=f'{i}{dst} = pto.vcvt {src}, {mask} {{part = "{part}"}} : !pto.vreg<128xbf16>, !pto.mask<b16> -> !pto.vreg<64xf32>'
  out.append(line)
 assert n==6 and not maps,(n,maps)
 s='\n'.join(out)+'\n'
 pat=r'(?m)^(\s*)(%\w+) = pto.vcvt ([^\n]+)\n\s*(%\w+) = pto.vcvt ([^\n]+)\n\s*pto.vsts \2, (%\w+)\[(%\w+)\], (%\w+) \{dist = "PK_B32"\}[^\n]+\n\s*pto.vsts \4,[^\n]+'
 def narrow(m):
  i,a,ac,b,bc,ptr,off,m32=m.groups();bc=bc.replace('part = "EVEN"','part = "ODD"')
  return f'{i}{a} = pto.vcvt {ac}\n{i}{b} = pto.vcvt {bc}\n{i}%layout_merge = pto.vor {a}, {b}, {mask} : !pto.vreg<128xbf16>, !pto.vreg<128xbf16>, !pto.mask<b16> -> !pto.vreg<128xbf16>\n{i}pto.vsts %layout_merge, {ptr}[{off}], {mask} : !pto.vreg<128xbf16>, !pto.ptr<bf16, ub>, !pto.mask<b16>'
 s,n=re.subn(pat,narrow,s);assert n==1,n
 ptr=re.search(r'pto.vsts %\w+#0, (%\w+)\[',s)[1];limit=re.search(r'(%\w+) = arith.constant '+str(h//128)+r' : i16',s)[1]
 restore=f'''      pto.vecscope {{
        %canonical_mask = pto.pset_b32 "PAT_ALL" : !pto.mask<b32>
        scf.for %canonical_pair = %c0_i16 to {limit} step %c1_i16 : i16 {{
          %canonical_i = arith.index_cast %canonical_pair : i16 to index
          %canonical_128 = arith.constant 128 : index
          %canonical_64 = arith.constant 64 : index
          %canonical_off = arith.muli %canonical_i, %canonical_128 : index
          %canonical_hi = arith.addi %canonical_off, %canonical_64 : index
          %canonical_even = pto.vlds {ptr}[%canonical_off] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
          %canonical_odd = pto.vlds {ptr}[%canonical_hi] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
          %canonical_a, %canonical_b = pto.vintlv %canonical_even, %canonical_odd : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
          pto.vsts %canonical_a, {ptr}[%canonical_off], %canonical_mask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
          pto.vsts %canonical_b, {ptr}[%canonical_hi], %canonical_mask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
        }}
      }}
'''
 needle='      pto.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID5>]';assert s.count(needle)==1;s=s.replace(needle,restore+needle)
 (D/f'{case}.layout.vpto').write_text(s)
 print(case,'patched')
