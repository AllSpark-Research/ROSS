# -*- coding: utf-8 -*-
"""Small final-answer literal gate, independent of test-pass summaries."""
import argparse,concurrent.futures,json,re,shutil,time
from pathlib import Path
import compat
r=compat.r;p=r.p

INSTRUCTIONS='''FINAL ANSWER FACT CHECK: independently compare EVERY listed inline
literal to original task or actual working-tree observations. The earlier draft
is fallible. A test-pass log does not establish that a prose path, date format,
filename, method, or field is correct. Inspect the code expression that generates
the claimed value and compare separators and directory levels literally.
Example: report logs/YYYY-M/D/ versus code f"{ROOT}/{year}-{month}-{day}" has a
different directory hierarchy and is CONTRADICTED even if all tests pass and an
older reference commit used another representation. Cite the last observation
containing the actual expression, not merely the last test/stat summary.
Do not assume the provided evidence index's 'current' label supersedes chronology.
Also recheck narrative factual claims outside backticks. Preserve valid names,
tests and adjacent explanations. Return the full replacement step row under the
standard schema, with factual revisions for every prior defect, plus:
"literal_checks":[{"id":"G040:B001","relation":"matches|contradicts|unknown|not_a_claim",
"observed_ref":"OBS_G031:L0055","reason":"explicit comparison of literal and actual evidence"}]
Cover every literal target exactly once. observed_ref may be null only for unknown
or not_a_claim; otherwise it must point to TASK/OBS raw evidence. A contradicted
literal needs a local defect span and a contradicted narrative line. Do not mask
the whole sentence when only its path/format is wrong. Missing evidence means
unknown, never silently supported. All literal comparison records are returned
by this model call, not precomputed correctness labels.'''

def targets(state,cat,sid):
    out=[]
    for ref in cat['checks'][sid]['narrative_lines']:
        line=cat['refs'][ref]
        for m in re.finditer(r'(?<!`)`([^`\n]+)`(?!`)',line['text']):
            out.append({'id':sid+':B%03d'%(len(out)+1),'line_ref':ref,'literal':m.group(1),'start':line['start']+m.start(1),'end':line['start']+m.end(1)})
    return out

def validate_literals(row,state,cat):
    wanted=targets(state,cat,row['id']);r.coverage(row.get('literal_checks'),'id',[x['id'] for x in wanted]);lookup={x['id']:x for x in wanted};checks={x['ref']:x for x in row['narrative_checks']}
    spans=[r.locate(cat,row['id'],s) for d in row['defects'] for s in d['spans']]
    for c in row['literal_checks']:
        if c.get('relation') not in ['matches','contradicts','unknown','not_a_claim'] or not c.get('reason'):raise ValueError('Literal comparison relation/reason required')
        ref=c.get('observed_ref')
        if c['relation'] in ['matches','contradicts']:
            r.evidence(cat,[ref]);source=cat['refs'][ref]['source']
            if source.startswith('G'):raise ValueError('A final claim cannot prove itself or cite another assistant assertion as proof')
        if c['relation']=='contradicts':
            target=lookup[c['id']]
            if not any(r.p.overlap((target['start'],target['end']),s) for s in spans):raise ValueError('Contradicted literal not locally excluded')
            if checks[target['line_ref']]['verdict']!='contradicted':raise ValueError('Narrative and literal comparison disagree')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--source',type=Path,required=True);ap.add_argument('--tokenizer',type=Path,required=True);a=ap.parse_args();root=a.root;started=time.time()
    if not (root/'before_literal_STATUS.json').exists():
        assert p.read(root/'STATUS.json')['complete'];shutil.copy2(root/'STATUS.json',root/'before_literal_STATUS.json');shutil.copytree(root/'final',root/'before_literal_final')
    states={};cats={};draft={};jobs=[]
    for f in (root/'prepared').glob('*/state.json'):
        s=p.read(f);n=s['sample'];states[n]=s;cats[n]=r.catalog(s);draft[n]=p.read(root/'review'/(n+'.json'))
        for step in s['steps']:
            if not step['calls']:jobs.append((n,step['id']))
    original=r.validate
    def validate(row,state,cat,prior=None):original(row,state,cat,prior);validate_literals(row,state,cat)
    r.validate=validate
    client=r.Client(root,p.read(root/'config.json')['endpoint']);results={n:{} for n in states};errors=[]
    def job(item):
        n,sid=item;extra={'final_literal_instructions':INSTRUCTIONS,'literal_targets':targets(states[n],cats[n],sid)}
        return client.rows('review',states[n],cats[n],[sid],extra,root/'calls'/n/'final'/sid,draft[n])
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        fs={pool.submit(job,j):j for j in jobs}
        for f in concurrent.futures.as_completed(fs):
            n,sid=fs[f]
            try:results[n].update(f.result())
            except Exception as exc:errors.append({'stage':'final_literal','sample':n,'step':sid,'error':str(exc)})
    for n,s in states.items():
        p.dump(root/'literal_results'/(n+'.json'),results[n])
        if results[n]:
            merged=dict(draft[n],**results[n]);r.compile_mask(s,cats[n],[merged[st['id']] for st in s['steps']],a.source/n,a.tokenizer,root/'final'/n)
    status=p.read(root/'before_literal_STATUS.json');status.update(complete=not errors,errors=errors,calls=client.count,final_literal_steps=sum(len(x) for x in results.values()),final_literal_seconds=time.time()-started,completed_at=time.time())
    p.dump(root/'literal_status.json',{'started_at':started,'completed_at':time.time(),'seconds':time.time()-started,'errors':errors,'steps':sum(len(x) for x in results.values())});p.dump(root/'STATUS.json',status)

if __name__=='__main__':main()
