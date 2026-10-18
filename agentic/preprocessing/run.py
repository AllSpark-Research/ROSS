# -*- coding: utf-8 -*-
"""Source anchors, exhaustive step checks, omissions, and factual rereview."""
import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import threading
import time
import urllib.request
import pipeline as p

HERE=Path(__file__).resolve().parent

def catalog(state):
    refs={}; checks={}; rendered=[]
    steps={x['id']:x for x in state['steps']}
    for sid,text in state['sources'].items():
        lines=[];offset=0
        for i,line in enumerate(text.splitlines(keepends=True),1):
            clean=line.rstrip('\r\n');ref=f'{sid}:L{i:04d}'
            refs[ref]={'source':sid,'start':offset,'end':offset+len(clean),'text':clean}
            lines.append((ref,clean));offset+=len(line)
        rendered.append('BEGIN SOURCE '+sid+'\n'+'\n'.join(k.split(':')[1]+'|'+v for k,v in lines)+'\nEND SOURCE '+sid)
        if sid not in steps:continue
        st=steps[sid];narrative=[];calls=[];quotes=[]
        for ref,line in lines:
            item=refs[ref]
            if not p.PROTOCOL.sub('',line).strip():continue
            if any(part['kind']=='thought_or_text' and max(part['start'],item['start'])<min(part['end'],item['end']) for part in st['parts']):
                narrative.append(ref)
        for i,part in enumerate([x for x in st['parts'] if x['kind']=='tool_call'],1):
            calls.append({'id':f'{sid}:C{i:02d}','name':part['tool_name'],'start':part['start'],'end':part['end']})
        for i,m in enumerate(re.finditer(r'"""|\x27\x27\x27',text),1):
            ref=f'{sid}:Q{i:04d}';line_ref=next(k for k,_ in lines if refs[k]['start']<=m.start()<refs[k]['end'])
            refs[ref]={'source':sid,'start':m.start(),'end':m.end(),'text':m.group(),'line_ref':line_ref}
            quotes.append({'ref':ref,'line':line_ref,'text':m.group(),'context':text[max(0,m.start()-45):m.end()+45]})
        checks[sid]={'narrative_lines':narrative,'calls':calls,'quote_anchors':quotes}
    return {'refs':refs,'checks':checks,'full_context':'\n\n'.join(rendered)}

def locate(cat,sid,span):
    if not isinstance(span,dict) or span.get('ref') not in cat['refs']:raise ValueError('Unknown span ref '+str(span))
    ref=cat['refs'][span['ref']]
    if ref['source']!=sid:raise ValueError('Span outside owned step')
    if 'quote' not in span:return ref['start'],ref['end']
    q=span['quote'];occ=span.get('occurrence',0)
    if not isinstance(q,str) or not q or type(occ)!=int or occ<0:raise ValueError('Exact nonempty substring and integer occurrence required')
    hits=p.positions(ref['text'],q)
    if occ>=len(hits):raise ValueError('Substring absent in '+span['ref']+'; exact original line is '+repr(ref['text']))
    return ref['start']+hits[occ],ref['start']+hits[occ]+len(q)

def evidence(cat,items,required=True):
    if not isinstance(items,list) or required and not items:raise ValueError('Evidence line refs required')
    for ref in items:
        if not isinstance(ref,str) or ref not in cat['refs'] or ':L' not in ref:raise ValueError('Unknown evidence line '+str(ref))

def ground_validate(obj,cat):
    if not isinstance(obj,dict) or not isinstance(obj.get('facts'),list):raise ValueError('facts array required')
    seen=set()
    for f in obj['facts']:
        if not f.get('id') or f['id'] in seen or not f.get('statement'):raise ValueError('Unique fact id and statement required')
        seen.add(f['id']);evidence(cat,f.get('evidence'))
        if any(cat['refs'][x]['source'].startswith('G') for x in f['evidence']):raise ValueError('Grounding cannot cite assistant claim as fact')

def coverage(items,key,expected):
    if not isinstance(items,list) or any(not isinstance(x,dict) for x in items):raise ValueError(key+' row array required')
    actual=[x.get(key) for x in items]
    if len(actual)!=len(set(actual)) or set(actual)!=set(expected):raise ValueError('Coverage mismatch '+key+'; missing='+str(sorted(set(expected)-set(actual)))+'; unexpected='+str(set(actual)-set(expected)))

def validate(row,state,cat,prior=None):
    sid=row.get('id');c=cat['checks'][sid]
    for key in ['defects','omissions','uncertain','revisions']:
        if not isinstance(row.get(key),list):raise ValueError(key+' array required')
    if not row.get('summary') or not row.get('delimiter_review'):raise ValueError('Summary and delimiter coverage account required')
    if any(k in row for k in ['value','impact','step_mask']):raise ValueError('Value cannot change factual masks')
    coverage(row.get('narrative_checks'),'ref',c['narrative_lines'])
    coverage(row.get('call_checks'),'id',[x['id'] for x in c['calls']])
    coverage(row.get('quote_checks'),'ref',[x['ref'] for x in c['quote_anchors']])
    spans=[];preserve=[];dids=set()
    for d in row['defects']:
        if not re.fullmatch(re.escape(sid)+r'\.D\d+',d.get('id','')) or d['id'] in dids:raise ValueError('Unique owned defect ID required')
        dids.add(d['id'])
        if d.get('kind') not in ['assertion','argument','expression','delimiter','patch_format','code_logic'] or not d.get('reason'):raise ValueError('Present defect kind/reason required; omissions belong separately')
        evidence(cat,d.get('evidence'))
        if not isinstance(d.get('spans'),list) or not d['spans'] or not isinstance(d.get('preserve'),list):raise ValueError('Present defect needs spans and preserve array')
        spans += [locate(cat,sid,x) for x in d['spans']]
        preserve += [locate(cat,sid,x) for x in d['preserve']]
    for x in row['uncertain']:
        if not x.get('reason'):raise ValueError('Uncertainty reason required')
        preserve.append(locate(cat,sid,x))
    oids=set()
    for o in row['omissions']:
        if not re.fullmatch(re.escape(sid)+r'\.O\d+',o.get('id','')) or o['id'] in oids:raise ValueError('Unique omission id required')
        oids.add(o['id'])
        if 'spans' in o or 'quote' in o or o.get('side') not in ['before','after'] or not o.get('missing') or not o.get('reason'):raise ValueError('Omission has a missing item/location, NEVER a span')
        evidence(cat,o.get('evidence'));omission_line=locate(cat,sid,{'ref':o['at_ref']})
        if 'preserve' in o or 'independent_defects' in o:
            if not isinstance(o.get('preserve'),list) or not o['preserve'] or not isinstance(o.get('independent_defects'),list) or not o['independent_defects']:raise ValueError('Mixed omission needs explicit preserved substrings AND independent present-defect IDs')
            for did in o['independent_defects']:
                if did not in dids:raise ValueError('Mixed omission refers to missing independent defect')
                defect=next(x for x in row['defects'] if x['id']==did)
                if not any(p.overlap(omission_line,locate(cat,sid,x)) for x in defect['spans']):raise ValueError('Independent defect is not on the mixed omission line')
            for keep in o['preserve']:
                lo,hi=locate(cat,sid,keep)
                if not omission_line[0]<=lo<hi<=omission_line[1]:raise ValueError('Mixed omission preservation must be within its anchor line')
                preserve.append((lo,hi))
        else:preserve.append(omission_line)
    protected=next(x['protected'] for x in state['steps'] if x['id']==sid)
    if any(a>=b for a,b in spans):raise ValueError('Empty span')
    if any(p.overlap(a,b) for i,a in enumerate(spans) for b in spans[i+1:]):raise ValueError('Overlapping exclusion spans')
    if any(p.overlap(a,b) for a in spans for b in preserve+protected):raise ValueError('Exclude overlaps preserved/omission-adjacent/uncertain/protocol content. For a genuinely mixed line only: omission may add preserve:[span selectors] for its valid substrings and independent_defects:[existing defect IDs] for independently wrong PRESENT substrings on that line. Pure missing-marker events still protect the whole valid adjacent line. Never use correct payload as a proxy for absent content.')
    for q in row['quote_checks']:
        if q.get('role') not in ['payload','outer','valid_inner','conflict','uncertain']:raise ValueError('Quote role required')
        lo,hi=locate(cat,sid,{'ref':q['ref']});covered=all(any(a<=i<b for a,b in spans) for i in range(lo,hi))
        if q['role']=='conflict' and not covered:raise ValueError('Every conflicting delimiter must be selected: '+q['ref'])
        if q['role']!='conflict' and any(p.overlap((lo,hi),x) for x in spans):raise ValueError('Cannot mask a delimiter simultaneously labeled nonconflicting: '+q['ref'])
    for n in row['narrative_checks']:
        if n.get('verdict') not in ['supported','exploration','contradicted','uncertain','nonclaim']:raise ValueError('Narrative verdict required')
        evidence(cat,n.get('evidence'),n['verdict'] in ['supported','contradicted'])
        if n['verdict']=='contradicted' and not any(p.overlap(locate(cat,sid,{'ref':n['ref']}),x) for x in spans):raise ValueError('Contradicted narrative line has no local defect: '+n['ref'])
    for call in row['call_checks']:
        if call.get('verdict') not in ['valid','defective','omission','uncertain'] or not call.get('reason'):raise ValueError('Call verdict/reason required')
    old={x['id']:x for x in prior['defects']} if prior else {}
    coverage(row['revisions'],'id',old)
    for x in row['revisions']:
        if not x.get('reason') or x.get('action') not in ['retain','refine','disprove','omission']:raise ValueError('Explicit factual revision required')
        evidence(cat,x.get('evidence'))
        if (x['action'] in ['retain','refine'])!=(x['id'] in dids):raise ValueError('Revision action contradicts retained defect ID')
        if x['action']=='omission' and not row['omissions']:raise ValueError('Omission revision needs actual omission record')

def parse(text,key):
    text=text.strip()
    try:obj=json.loads(text)
    except ValueError:
        blocks=re.findall(r'```(?:json)?\s*\n?(.*?)\n?```',text,re.S)
        if len(blocks)!=1:raise ValueError('One complete JSON result required')
        obj=json.loads(blocks[0])
    if isinstance(obj,list):
        if all(isinstance(x,dict) and set(x)=={key} and isinstance(x[key],list) for x in obj):obj={key:[y for x in obj for y in x[key]]}
        else:obj={key:obj}
    if not isinstance(obj,dict) or not isinstance(obj.get(key),list) or any(not isinstance(x,dict) for x in obj[key]):raise ValueError('Invalid result container')
    return obj

class Client:
    def __init__(self,root,endpoint):self.root=root;self.endpoint=endpoint;self.lock=threading.Lock();self.count=len(list((root/'calls').rglob('request.json')))
    def call(self,stage,state,cat,ids,extra,dest):
        prompt=(HERE/'ground.txt').read_text() if stage=='ground' else ((HERE/'review.txt').read_text() if stage=='review' else '')+(HERE/'rules.txt').read_text()
        focus=[]
        for sid in ids:
            pos=[x['id'] for x in state['steps']].index(sid)
            previous='OBS_'+state['steps'][pos-1]['id'] if pos else 'TASK'
            for source in [previous,sid,'OBS_'+sid]:
                if source not in state['sources']:continue
                focus.append('FOCUS SOURCE '+source+'\n'+'\n'.join(k.split(':')[1]+'|'+v['text'] for k,v in cat['refs'].items() if v['source']==source and ':L' in k))
        packet=extra if stage=='ground' else {'owned_steps':ids,'catalog':{sid:cat['checks'][sid] for sid in ids},**extra}
        ending='OUTPUT THE EVIDENCE INDEX ONLY' if stage=='ground' else 'OUTPUT ONLY OWNED STEPS '+json.dumps(ids)
        user=cat['full_context']+'\n\n'+json.dumps(packet,ensure_ascii=False)+'\n\n'+'\n'.join(focus)+'\n'+ending
        body={'model':os.environ.get('ROSS_MODEL','judge-model'),'messages':[{'role':'system','content':prompt},{'role':'user','content':user}],'reasoning_effort':'high','chat_template_kwargs':{'enable_thinking':True},'temperature':0,'max_tokens':32768,'stream':False}
        fingerprint=p.digest({'body':body,'endpoint':self.endpoint});cache=dest/'cache.json'
        if cache.exists():
            x=p.read(cache)
            if x['fingerprint']!=fingerprint:raise ValueError('Cached input changed; use a fresh experiment')
            return p.read(dest/'response.json')
        with self.lock:
            if self.count>=120:raise RuntimeError('configured call cap reached')
            self.count+=1
        p.dump(dest/'request.json',body);started=time.time()
        headers={'Content-Type':'application/json'}
        key=os.environ.get('ROSS_API_KEY','')
        if key:
            name=os.environ.get('ROSS_AUTH_HEADER','Authorization');scheme=os.environ.get('ROSS_AUTH_SCHEME','Bearer')
            headers[name]=f'{scheme} {key}'.strip()
        req=urllib.request.Request(self.endpoint,data=json.dumps(body).encode(),headers=headers)
        with urllib.request.urlopen(req,timeout=600) as response:raw=json.load(response)
        p.dump(dest/'response.json',raw);p.dump(cache,{'fingerprint':fingerprint,'seconds':time.time()-started,'response_sha256':p.digest(raw)})
        return raw
    def rows(self,stage,state,cat,ids,extra,dest,prior=None,depth=0):
        # Repair packets contain only their owned draft rows/catalog, never siblings.
        if prior:extra=dict(extra,draft=[prior[s] for s in ids])
        raw=self.call(stage,state,cat,ids,extra,dest);valid={};errors=[]
        try:
            choice=raw['choices'][0]
            if choice['finish_reason']!='stop':raise ValueError('Incomplete response '+choice['finish_reason'])
            obj=parse(choice['message'].get('content') or '', 'steps')
            counts={sid:sum(x.get('id')==sid for x in obj['steps']) for sid in ids}
            for row in obj['steps']:
                sid=row.get('id')
                if sid not in ids or counts[sid]!=1:continue
                try:validate(row,state,cat,prior.get(sid) if prior else None);valid[sid]=row
                except (ValueError,KeyError,TypeError,AttributeError,IndexError) as exc:errors.append({'id':sid,'error':str(exc),'unvalidated_output':row})
        except (ValueError,KeyError,TypeError,AttributeError,IndexError) as exc:errors.append({'error':str(exc)})
        missing=[s for s in ids if s not in valid];p.dump(dest/'validation.json',{'valid':list(valid),'missing':missing,'errors':errors})
        if missing:
            if depth>=2:raise ValueError('Unresolved rows '+str(missing))
            for sid in missing:
                packet=dict(extra,validation_feedback=[e for e in errors if e.get('id') in [None,sid]],reminder='Use exact owned source refs. Correct source/schema errors; never invent semantic labels to pass validation.')
                valid.update(self.rows(stage,state,cat,[sid],packet,dest/('repair_'+sid),prior,depth+1))
        p.dump(dest/'accepted.json',{'steps':[valid[s] for s in ids]});return valid

def combine(state,cat,rows):
    spans=[];unknown=[];omissions=[]
    for row in rows:
        sid=row['id']
        for d in row['defects']:
            for s in d['spans']:
                lo,hi=locate(cat,sid,s)
                spans.append({'region_id':sid,'char_start':lo,'char_end':hi,'resolved_quote':state['sources'][sid][lo:hi],
                              'defect_id':d['id'],'rationale':d['reason'],'evidence':[{'source_ref':x,'source_id':cat['refs'][x]['source'],'quote':cat['refs'][x]['text']} for x in d['evidence']]})
        for x in row['uncertain']:
            lo,hi=locate(cat,sid,x);q=state['sources'][sid][lo:hi]
            unknown.append({'step_id':sid,'quote':q,'occurrence':p.positions(state['sources'][sid],q).index(lo),'reason':x['reason']})
        omissions.extend(dict(x,step_id=sid,mask_action='none; absent content cannot target adjacent valid tokens') for x in row['omissions'])
    return spans,unknown,omissions

