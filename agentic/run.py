# -*- coding: utf-8 -*-
"""Batch semantic review, defect localization, and validation for agentic trajectories."""
import argparse,json,os,sys,time,threading,traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
HERE=Path(__file__).resolve().parent
ANNOTATION=HERE/'annotation'
PREPROCESSING=HERE/'preprocessing'
sys.path.insert(0,str(ANNOTATION)); sys.path.insert(0,str(PREPROCESSING))
import experiment as X
import run as base
if os.environ.get('ROSS_ANCHOR_COMPAT','1')=='1':
    import anchor_compat            # Align call-level anchors with downstream boundary validation.
REPAIR_LOG=[]
if os.environ.get('ROSS_REPAIR','1')=='1':
    import repair as _repair
    _orig_vs=X.validate_semantic
    def _vs(row,state,cat):
        try:
            a=_repair.repair_semantic(row,cat)
            if a: REPAIR_LOG.extend(a)
        except Exception:
            pass
        return _orig_vs(row,state,cat)
    X.validate_semantic=_vs
from client import Client,dump,MODEL,ENDPOINT
p=base.p
lock=threading.Lock(); done=[0,0,0]   # windows_done, steps_ok, failures

def process(sample_dir,root,max_tokens):
    name=sample_dir.name
    out_sem=root/'semantic'/(name+'.json'); out_loc=root/'localize'/(name+'.json')
    ann=root/'final'/name/'annotation.json'
    if ann.exists():
        try:
            if json.loads(ann.read_text()).get('coverage',{}).get('complete'): return ('skip',name,0,0)
        except Exception: pass
    elif out_loc.exists() and os.environ.get('ROSS_RETRY','0')!='1':
        return ('skip',name,0,0)
    st=json.loads((sample_dir/'state.json').read_text())
    cat=base.catalog(st)
    client=Client(root/'calls'/name,max_calls=int(os.environ.get('ROSS_MAX_CALLS','400')))
    sem=json.loads(out_sem.read_text()) if out_sem.exists() else {}
    wins=list(p.windows(st,6,18000))
    # Revalidate cached responses before issuing retry calls.
    if os.environ.get('ROSS_RETRY','0')=='1':
        import glob as _g
        need={x['id'] for x in st['steps']} - set(sem)
        if need:
            for rf in sorted(_g.glob(str(root/'calls'/name/'semantic'/'*'/'*'/'response.json'))):
                if not need: break
                try: rows=X.parse(json.loads(open(rf).read()))
                except Exception: continue
                for row in rows:
                    if not isinstance(row,dict): continue
                    sid=row.get('id')
                    if sid not in need: continue
                    try:
                        X.validate_semantic(row,st,cat); sem[sid]=row; need.discard(sid)
                        with lock: done[1]+=0
                    except Exception: pass
            dump(out_sem,sem)
    for i,ids in enumerate(wins):
        todo=[s for s in ids if s not in sem]
        if not todo: continue
        try:
            sub='semantic_retry' if os.environ.get('ROSS_RETRY','0')=='1' else 'semantic'
            sem.update(X.run_window(client,'semantic',st,cat,todo,root/'calls'/name/sub/f'{i:03d}'))
        except Exception as e:
            with lock: done[2]+=1
        with lock: done[0]+=1
    dump(out_sem,sem)
    loc={}
    for sid in [x['id'] for x in st['steps']]:
        if sid not in sem: continue
        if not sem[sid]['errors'] and not sem[sid]['omissions']:
            loc[sid]={'id':sid,'defects':[],'omissions':[],'unresolved':[]}
    for i,ids in enumerate(wins):
        todo=[s for s in ids if s in sem and s not in loc]
        if not todo: continue
        try:
            sub='localize_retry' if os.environ.get('ROSS_RETRY','0')=='1' else 'localize'
            loc.update(X.run_window(client,'localize',st,cat,todo,root/'calls'/name/sub/f'{i:03d}',sem))
        except Exception as e:
            with lock: done[2]+=1
    dump(out_loc,loc)
    # Assemble the final annotation and record every unlabeled step explicitly.
    all_ids=[x['id'] for x in st['steps']]
    labeled=[i for i in all_ids if i in loc and i in sem]
    dropped=[i for i in all_ids if i not in labeled]
    try:
        rows=[X.resolve(loc[i],sem[i],st,cat) for i in labeled]
        spans,_,omissions=base.combine(st,cat,rows)
        dump(root/'final'/name/'annotation.json',
             {'coverage':{'expected_steps':len(all_ids),'labeled_steps':len(labeled),
                          'dropped_steps':len(dropped),
                          'coverage_ratio':round(len(labeled)/max(1,len(all_ids)),4),
                          'complete':len(dropped)==0},
              'unlabeled_steps':[{'id':i,
                                  'reason':'semantic validation failed after retries' if i not in sem
                                           else 'localization failed after retries'} for i in dropped],
              'steps':rows,'mask_spans':spans,'omissions':omissions,
              'semantic_uncertain':[dict(u,step_id=r['id']) for r in rows for u in r['semantic']['uncertain']],
              'localization_unresolved':[dict(u,step_id=r['id']) for r in rows for u in r['localization_unresolved']],
              'repair_audit_count':len(REPAIR_LOG)})
    except Exception as e:
        dump(root/'final'/name/'ERROR.json',{'error':str(e)[:300],
             'expected_steps':len(all_ids),'labeled_steps':len(labeled)})
    with lock: done[1]+=len(sem)
    return ('ok',name,len(sem),len(loc))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--prepared',type=Path,required=True)
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--workers',type=int,default=16)
    ap.add_argument('--limit',type=int,default=0)
    ap.add_argument('--max-tokens',type=int,default=24576)
    a=ap.parse_args()
    os.environ['ROSS_MAXTOK']=str(a.max_tokens)
    samples=sorted([d for d in a.prepared.iterdir() if d.is_dir() and (d/'state.json').exists()])
    if a.limit: samples=samples[:a.limit]
    a.root.mkdir(parents=True,exist_ok=True)
    dump(a.root/'config.json',{'model':MODEL,'endpoint':ENDPOINT,'samples':len(samples),
                               'anchor_compat':os.environ.get('ROSS_ANCHOR_COMPAT','1'),
                               'note':'final semantic-review and localization protocol'})
    print(f"samples={len(samples)} workers={a.workers} model={MODEL}",flush=True)
    t0=time.time(); n=0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for r in ex.map(lambda d: _safe(d,a.root,a.max_tokens), samples):
            n+=1
            if n%10==0:
                el=time.time()-t0
                print(f"  {n}/{len(samples)} samples | windows={done[0]} steps={done[1]} failures={done[2]} | {el/60:.1f}min",flush=True)
    print(f"DONE samples={n} windows={done[0]} steps={done[1]} failures={done[2]} elapsed={(time.time()-t0)/60:.1f}min",flush=True)

def _safe(d,root,mt):
    try: return process(d,root,mt)
    except Exception as e:
        with lock: done[2]+=1
        return ('err',d.name,0,0)
if __name__=='__main__':main()
