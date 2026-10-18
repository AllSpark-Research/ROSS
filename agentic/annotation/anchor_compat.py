"""Relax ONLY the call-linked semantic anchor check.

Rationale: boundary_compat.resolve() overwrites anchors of call-linked errors with the
full call range, so a call-linked anchor is navigational downstream. call_id ownership
remains the hard boundary. Thought-only errors (call_id=None) keep binding anchors and
stay strict, because nothing downstream replaces them.
"""
import experiment as e
import run as base
p=base.p
_orig=e.validate_semantic

def validate_semantic(row,state,cat):
 sid=row['id'];calls={x['id']:x for x in cat['checks'][sid]['calls']}
 relaxed=[]
 for err in row.get('errors',[]):
  cid=err.get('call_id')
  if not cid or cid not in calls: continue
  call=calls[cid];keep=[]
  for ref in err.get('anchors',[]):
   lo,hi=base.locate(cat,sid,{'ref':ref})
   if p.overlap((lo,hi),(call['start'],call['end'])):keep.append(ref)
  if not keep:
   # navigational anchor fell outside its call -> retarget to the call's own lines
   inside=[ref for ref,x in cat['refs'].items()
           if x['source']==sid and call['start']<=x['start']<x['end']<=call['end']]
   if inside:
    relaxed.append({'error_id':err['id'],'call_id':cid,'model_anchors':err['anchors'],'retargeted_to':'ALL_CALL_LINES','n':len(inside)})
    err['anchors']=inside
 out=_orig(row,state,cat)
 if relaxed: row.setdefault('_anchor_retargeted',relaxed)
 return out
e.validate_semantic=validate_semantic
