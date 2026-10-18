# -*- coding: utf-8 -*-
"""Semantic review with independent call dimensions, causal diagnosis, and local boundaries."""
import argparse,concurrent.futures,hashlib,importlib.util,json,os,re,sys,time
from pathlib import Path
from client import Client,dump,digest
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'preprocessing'))
import run as base
import compat
p=base.p
OLD=Path(os.environ.get('ROSS_PREPARED_ROOT', HERE.parent/'prepared'))

def enum(v,choices):
 if v not in choices:raise ValueError('Invalid enum '+str(v))
def validate_semantic(row,state,cat):
 sid=row['id'];c=cat['checks'][sid]
 if not row.get('summary'):raise ValueError('summary required')
 for key in ['errors','omissions','uncertain','causal_checks']:
  if not isinstance(row.get(key),list):raise ValueError(key+' array required')
 base.coverage(row.get('narrative_checks'),'ref',c['narrative_lines']);base.coverage(row.get('calls'),'id',[x['id'] for x in c['calls']]);base.coverage(row.get('quote_checks'),'ref',[x['ref'] for x in c['quote_anchors']])
 calls={x['id']:x for x in c['calls']};events={}
 for key,letter in [('errors','D'),('omissions','O')]:
  for e in row[key]:
   if not re.fullmatch(re.escape(sid)+r'\.'+letter+r'\d+',e.get('id','')) or e['id'] in events:raise ValueError('Unique owned event id required')
   base.evidence(cat,e.get('evidence'));cid=e.get('call_id');enum(cid,[None]+list(calls));events[e['id']]=e
   if key=='errors':
    enum(e.get('kind'),['assertion','argument','expression','delimiter','patch_format','code_logic'])
    if not e.get('diagnosis') or not e.get('anchors'):raise ValueError('diagnosis and anchors required')
    for ref in e['anchors']:
     lo,hi=base.locate(cat,sid,{'ref':ref})
     if cid and not p.overlap((lo,hi),(calls[cid]['start'],calls[cid]['end'])):raise ValueError('Error anchor not within its call')
   else:
    enum(e.get('side'),['before','after']);lo,hi=base.locate(cat,sid,{'ref':e['at_ref']})
    if not e.get('missing') or not e.get('reason'):raise ValueError('Omission missing/reason required')
    if cid and not p.overlap((lo,hi),(calls[cid]['start'],calls[cid]['end'])):raise ValueError('Omission anchor not in its call')
 for n in row['narrative_checks']:
  enum(n.get('verdict'),['supported','exploration','contradicted','uncertain','nonclaim']);base.evidence(cat,n.get('evidence'),n['verdict'] in ['supported','contradicted'])
  if n['verdict']=='contradicted' and not any(n['ref'] in e['anchors'] for e in row['errors']):raise ValueError('Contradicted line needs local error '+n['ref'])
 seen=set()
 for n in row['causal_checks']:
  if n.get('ref') not in c['narrative_lines'] or n['ref'] in seen:raise ValueError('Unique owned causal line required')
  seen.add(n['ref']);enum(n.get('verdict'),['supported','contradicted','hypothesis','unknown']);base.evidence(cat,n.get('evidence'),n['verdict'] in ['supported','contradicted'])
  if not n.get('claimed_cause') or not n.get('observed_layer'):raise ValueError('Causal witness required')
  if n['verdict']=='contradicted' and not any(e['id'] in n.get('error_ids',[]) and n['ref'] in e['anchors'] for e in row['errors']):raise ValueError('Contradicted causal claim needs local error')
 for call in row['calls']:
  for dim in ['contract','operation','environment']:
   d=call[dim];enum(d.get('verdict'),['pass','fail','unknown','not_applicable']);base.evidence(cat,d.get('evidence'),d['verdict']=='fail')
   if not d.get('reason') or not isinstance(d.get('error_ids'),list):raise ValueError('Dimension reason/error_ids required')
   for eid in d['error_ids']:
    if eid not in events or events[eid].get('call_id')!=call['id']:raise ValueError('Dimension links unrelated event')
   if d['verdict']=='fail' and not d['error_ids']:raise ValueError('FAIL needs its own present error/omission; API pass does not excuse operation fail')
  d=call['execution'];enum(d.get('verdict'),['succeeded','failed','not_executed','unknown']);base.evidence(cat,d.get('evidence'),d['verdict']!='unknown')
 for e in events.values():
  if e.get('call_id'):
   call=next(x for x in row['calls'] if x['id']==e['call_id'])
   if not any(e['id'] in call[k]['error_ids'] and call[k]['verdict']=='fail' for k in ['contract','operation','environment']):raise ValueError('Present event needs a matching failed dimension '+e['id'])
 for q in row['quote_checks']:
  enum(q.get('role'),['payload','outer','valid_inner','conflict','uncertain'])
  if q['role']=='conflict' and not any(q['ref'] in e['anchors'] for e in row['errors']):raise ValueError('Every conflict Q needs its exact error anchor')
 for u in row['uncertain']:
  base.locate(cat,sid,{'ref':u['ref']})
  if not u.get('reason'):raise ValueError('Uncertainty reason required')

def resolve(row,semantic,state,cat):
 sid=row['id'];events={x['id']:x for x in semantic['errors']+semantic['omissions']}
 actual=[x['id'] for key in ['defects','omissions','unresolved'] for x in row.get(key,[])]
 if len(actual)!=len(set(actual)) or set(actual)!=set(events):raise ValueError('Localize each semantic event exactly once; unresolved is allowed, no silent dropping')
 spans=[];preserve=[];defects=[];omissions=[]
 for d in row['defects']:
  e=events[d['id']]
  if 'diagnosis' not in e or not d.get('spans') or not isinstance(d.get('preserve'),list):raise ValueError('Present error needs spans and preserve')
  for x in d['spans']:
   lo,hi=base.locate(cat,sid,x)
   anchor_ranges=[base.locate(cat,sid,{'ref':ref}) for ref in e['anchors']]
   if not any(a<=lo<hi<=b for a,b in anchor_ranges):raise ValueError('Selector outside semantic error anchors')
   spans.append((lo,hi))
  preserve.extend(base.locate(cat,sid,x) for x in d['preserve']);defects.append(dict(d,kind=e['kind'],reason=e['diagnosis'],evidence=e['evidence']))
 for o in row['omissions']:
  e=events[o['id']]
  if 'at_ref' not in e:raise ValueError('Wrong event type')
  ol=base.locate(cat,sid,{'ref':e['at_ref']});item=dict(e,**o)
  if 'preserve' in o or 'independent_defects' in o:
   if not o.get('preserve') or not o.get('independent_defects'):raise ValueError('Mixed omission needs both preserve and independent_defects')
   for did in o['independent_defects']:
    selected=next((d for d in defects if d['id']==did),None)
    if not selected or not any(p.overlap(ol,base.locate(cat,sid,x)) for x in selected['spans']):raise ValueError('Mixed omission needs actual same-line independent defect')
   for x in o['preserve']:
    lo,hi=base.locate(cat,sid,x)
    if not ol[0]<=lo<hi<=ol[1]:raise ValueError('Omission preservation must belong to adjacent line')
    preserve.append((lo,hi))
  else:preserve.append(ol)
  omissions.append(item)
 protected=next(x['protected'] for x in state['steps'] if x['id']==sid)
 if any(a>=b for a,b in spans) or any(p.overlap(a,b) for i,a in enumerate(spans) for b in spans[i+1:]):raise ValueError('Empty/overlapping exclusions')
 if any(p.overlap(a,b) for a in spans for b in preserve+protected):raise ValueError('Exclusion overlaps preserved/omission-adjacent/protocol source; refine exact responsibility or mark unresolved, never invent errors')
 unresolved_ids={x['id'] for x in row['unresolved']}
 for x in row['unresolved']:
  if not x.get('reason'):raise ValueError('Unresolved explanation required')
 for q in semantic['quote_checks']:
  lo,hi=base.locate(cat,sid,{'ref':q['ref']});covered=all(any(a<=i<b for a,b in spans) for i in range(lo,hi))
  unresolved=any(e['id'] in unresolved_ids and q['ref'] in e['anchors'] for e in semantic['errors'])
  if q['role']=='conflict' and not covered and not unresolved:raise ValueError('Missed confirmed conflict '+q['ref'])
  if q['role']!='conflict' and any(p.overlap((lo,hi),x) for x in spans):raise ValueError('Excluding nonconflicting quote '+q['ref'])
 return {'id':sid,'summary':semantic['summary'],'defects':defects,'omissions':omissions,'uncertain':[], 'semantic':semantic,'localization_unresolved':row['unresolved']}

def parse(raw):
 ch=raw['choices'][0]
 if ch['finish_reason']!='stop':raise ValueError('Output incomplete: '+str(ch['finish_reason']))
 return base.parse(ch['message'].get('content') or '', 'steps')['steps']

def packet(stage,state,cat,ids,semantic=None):
 if stage=='semantic':
  focus=[]
  for sid in ids:
   focus.append('FOCUS '+sid+'\n'+state['sources'][sid]+'\n'+state['sources'].get('OBS_'+sid,''))
  return cat['full_context']+'\nCATALOG\n'+json.dumps({sid:cat['checks'][sid] for sid in ids},ensure_ascii=False)+'\n'+'\n'.join(focus)+'\nOWNED STEPS '+json.dumps(ids)
 # Only original owned source + cited evidence neighborhoods, no rewritten model context summary.
 refs=set()
 for sid in ids:
  for e in semantic[sid]['errors']+semantic[sid]['omissions']:refs.update(e['evidence'])
 expanded=set(refs)
 for ref in refs:
  source,line=ref.rsplit(':L',1)
  expanded.update(f'{source}:L{i:04d}' for i in range(max(1,int(line)-2),int(line)+3) if f'{source}:L{i:04d}' in cat['refs'])
 source='\n\n'.join('SOURCE '+sid+'\n'+'\n'.join(k+'|'+v['text'] for k,v in cat['refs'].items() if v['source']==sid and ':L' in k) for sid in ids)
 evidence='\n'.join(ref+'|'+cat['refs'][ref]['text'] for ref in sorted(expanded))
 return source+'\nORIGINAL EVIDENCE\n'+evidence+'\nQ ANCHORS\n'+json.dumps({sid:cat['checks'][sid]['quote_anchors'] for sid in ids},ensure_ascii=False)+'\nSEMANTIC PROPOSALS\n'+json.dumps([semantic[sid] for sid in ids],ensure_ascii=False)+'\nOWNED STEPS '+json.dumps(ids)

def run_window(client,stage,state,cat,ids,dest,semantic=None):
 valid={};feedback=[]
 for attempt in range(3):
  missing=[s for s in ids if s not in valid]
  if not missing:break
  user=packet(stage,state,cat,missing,semantic)
  if feedback:
   hint=('\nVALIDATION FEEDBACK: repair schema/source only; do not invent semantics.\n'
         'COMMON FIXES — apply only if the error matches:\n'
         '- "Unknown evidence line": evidence must be a SINGLE existing line ref like OBS_G001:L0007. '
         'NEVER a range (no "L0016-L0220"), never invented refs. Copy refs verbatim from the CATALOG.\n'
         '- "Coverage mismatch ... unexpected": you emitted refs/ids outside the owned CATALOG. '
         'Emit EXACTLY the catalog entries for the owned steps, no more.\n'
         '- "Contradicted line needs local error": every narrative/causal line you mark "contradicted" '
         'must have an error whose anchors include that exact line ref.\n'
         '- "Dimension reason/error_ids required": every contract/operation/environment dimension needs '
         'both "reason" (string) and "error_ids" (list, may be empty).\n'
         '- "Present event needs a matching failed dimension": a call-linked error must be referenced by '
         'a dimension of that same call whose verdict is "fail".\n'
         '- "Output incomplete": emit more compact JSON; keep summaries/reasons short.\n')
   user+=hint+json.dumps(feedback,ensure_ascii=False)
  raw=client.call((HERE/'prompts'/('semantic.txt' if stage=='semantic' else 'localize.txt')).read_text(),user,dest/str(attempt));feedback=[]
  try:
   rows=parse(raw);counts={sid:sum(x.get('id')==sid for x in rows) for sid in missing}
   for row in rows:
    sid=row.get('id')
    if sid not in missing or counts[sid]!=1:continue
    try:
     if stage=='semantic':validate_semantic(row,state,cat)
     else:resolve(row,semantic[sid],state,cat)
     valid[sid]=row
    except (ValueError,KeyError,TypeError,AttributeError) as e:feedback.append({'id':sid,'error':str(e),'proposal':row})
   for sid in missing:
    if sid not in valid and not any(f.get('id')==sid for f in feedback):feedback.append({'id':sid,'error':'Missing or duplicated owned row'})
  except (ValueError,KeyError,TypeError) as e:feedback=[{'error':str(e)}]
  dump(dest/str(attempt)/'validation.json',{'valid':list(valid),'errors':feedback})
 dump(dest/'accepted.json',{'steps':[valid[s] for s in ids if s in valid],'missing':[s for s in ids if s not in valid]})
 return valid

