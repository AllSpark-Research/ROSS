# -*- coding: utf-8 -*-
"""Pre-validation repair with audited, semantics-preserving normalization.

The repair only normalizes references already present in model output:
1. an evidence line range is reduced to its first existing source line;
2. entries outside the owned catalog scope are removed.
It never creates an error or omission or changes a verdict, diagnosis, or span.
"""
import re
RANGE=re.compile(r'^([A-Za-z_]+[A-Za-z0-9_]*:L\d+)\s*[-–]\s*L?\d+$')

def _fix_ev(lst,refs,audit,where):
    if not isinstance(lst,list): return lst
    out=[]
    for e in lst:
        if not isinstance(e,str): continue
        if e in refs: out.append(e); continue
        m=RANGE.match(e.strip())
        if m and m.group(1) in refs:
            audit.append({'kind':'evidence_range_to_first','where':where,'from':e,'to':m.group(1)})
            out.append(m.group(1)); continue
        # Accept forms such as OBS_G002-L0016 (dash instead of colon).
        alt=e.replace('-L',':L',1)
        if alt in refs:
            audit.append({'kind':'evidence_dash_to_colon','where':where,'from':e,'to':alt})
            out.append(alt); continue
        audit.append({'kind':'evidence_dropped_unknown','where':where,'from':e})
    return out

def repair_semantic(row,cat):
    """Normalize one semantic-step row in place and return the audit trail."""
    audit=[]; sid=row.get('id')
    c=cat['checks'].get(sid)
    if not c: return audit
    refs=set(cat['refs'].keys())
    narr=set(c['narrative_lines']); callids={x['id'] for x in c['calls']}
    qrefs={x['ref'] for x in c['quote_anchors']}
    # 1) Normalize evidence references.
    for key in ('narrative_checks','causal_checks'):
        for x in row.get(key) or []:
            if isinstance(x,dict) and 'evidence' in x:
                x['evidence']=_fix_ev(x['evidence'],refs,audit,f'{sid}.{key}')
    for key in ('errors','omissions'):
        for x in row.get(key) or []:
            if isinstance(x,dict) and 'evidence' in x:
                x['evidence']=_fix_ev(x['evidence'],refs,audit,f'{sid}.{key}')
    for x in row.get('calls') or []:
        if not isinstance(x,dict): continue
        for dim in ('contract','operation','environment','execution'):
            d=x.get(dim)
            if isinstance(d,dict) and 'evidence' in d:
                d['evidence']=_fix_ev(d['evidence'],refs,audit,f'{sid}.{dim}')
    # 2) Remove out-of-scope entries; never synthesize missing entries.
    def prune(key,field,allowed):
        lst=row.get(key)
        if not isinstance(lst,list): return
        keep=[]
        for x in lst:
            v=x.get(field) if isinstance(x,dict) else None
            if v in allowed: keep.append(x)
            else: audit.append({'kind':'pruned_out_of_scope','where':f'{sid}.{key}','ref':v})
        row[key]=keep
    prune('narrative_checks','ref',narr)
    prune('causal_checks','ref',narr)
    prune('calls','id',callids)
    prune('quote_checks','ref',qrefs)
    return audit
