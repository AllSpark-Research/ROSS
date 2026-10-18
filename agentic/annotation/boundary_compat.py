"""Localization anchors navigate; call ownership remains the hard source boundary."""
import copy
import experiment as e
original=e.resolve

def resolve(row,semantic,state,cat):
 normalized=copy.deepcopy(row)
 allowed={x['id'] for x in semantic['errors']+semantic['omissions']}
 extras=[]
 for key in ['defects','omissions','unresolved']:
  extras.extend(x['id'] for x in normalized.get(key,[]) if x['id'] not in allowed)
  normalized[key]=[x for x in normalized.get(key,[]) if x['id'] in allowed]
 for d in normalized.get('defects',[]):d.setdefault('preserve',[])
 scoped=copy.deepcopy(semantic);calls={c['id']:c for c in cat['checks'][row['id']]['calls']}
 for err in scoped['errors']:
  cid=err.get('call_id')
  if cid:
   call=calls[cid]
   err['anchors']=[ref for ref,x in cat['refs'].items() if x['source']==row['id'] and call['start']<=x['start']<x['end']<=call['end']]
 result=original(normalized,scoped,state,cat);result['semantic']=semantic;result['out_of_scope_localization_ids']=extras
 return result

e.resolve=resolve
