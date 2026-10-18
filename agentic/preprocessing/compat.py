"""Resume with lossless support for a complete JSON after an unclosed code fence."""
import json,re
import run as r

original_parse=r.parse
original_rows=r.Client.rows

def diagnostics(row,cat):
    sid=row['id'];out=[]
    selected=[s for d in row.get('defects',[]) for s in d.get('spans',[])]
    protected=[('preserve',s) for d in row.get('defects',[]) for s in d.get('preserve',[])]+[('omission-adjacent',s) for o in row.get('omissions',[]) for s in o.get('preserve',[{'ref':o['at_ref']}])]+[('uncertain',s) for s in row.get('uncertain',[])]
    for span in selected:
        for kind,keep in protected:
            try:
                if not r.p.overlap(r.locate(cat,sid,span),r.locate(cat,sid,keep)):continue
                detail={'error':'same original characters both excluded and protected','selected':span,'protected_kind':kind,'protected':keep}
                if keep.get('quote') and keep.get('ref') in cat['refs']:
                    line=cat['refs'][keep['ref']]['text'];q=keep['quote']
                    detail['protected_quote_occurrences']=[{'occurrence':i,'context':line[max(0,j-35):j+len(q)+35]} for i,j in enumerate(r.p.positions(line,q))]
                out.append(detail)
            except (ValueError,KeyError,TypeError):pass
    return out

def parse(text,key):
    stripped=text.strip()
    if stripped.startswith('```') and stripped.count('```')==1:
        # Strip presentation prefix only. json.loads must consume a COMPLETE body.
        match=re.match(r'^```(?:json)?\s*\n',stripped)
        if match:
            candidate=stripped[match.end():]
            json.loads(candidate)
            return original_parse(candidate,key)
    return original_parse(text,key)

def rows(self,stage,state,cat,ids,extra,dest,prior=None,depth=0):
    if stage=='review':extra=dict(extra,mixed_omission_note='For a genuinely mixed line containing both an absent token and an independently wrong present substring, omission may carry preserve:[span selectors] and independent_defects:[existing defect IDs on that line]. Keep those explicit fields if still valid. Pure missing-marker events must not mask valid payload. No utility reasoning can erase factual errors.')
    if depth:return original_rows(self,stage,state,cat,ids,extra,dest,prior,depth)
    try:return original_rows(self,stage,state,cat,ids,extra,dest,prior,depth)
    except (ValueError,KeyError,TypeError,AttributeError,IndexError) as exc:
        valid={};invalid={}
        for f in sorted(dest.rglob('response.json'),key=lambda x:x.stat().st_mtime):
            raw=r.p.read(f)
            if raw['choices'][0]['finish_reason']!='stop':continue
            try:obj=parse(raw['choices'][0]['message'].get('content') or '', 'steps')
            except (ValueError,TypeError):continue
            for row in obj['steps']:
                sid=row.get('id')
                if sid not in ids:continue
                try:r.validate(row,state,cat,prior.get(sid) if prior else None);valid[sid]=row
                except (ValueError,KeyError,TypeError,AttributeError,IndexError) as err:invalid[sid]={'id':sid,'error':str(err),'unvalidated_output':row,'exact_conflicts':diagnostics(row,cat)}
        for sid in ids:
            if sid not in valid:
                packet=dict(extra,validation_feedback=[invalid.get(sid,{'id':sid,'error':str(exc)})],reminder='Repair exact source/schema errors only. Never invent labels to satisfy the validator.')
                valid.update(original_rows(self,stage,state,cat,[sid],packet,dest/'mixed_recovery'/sid,prior))
        r.p.dump(dest/'mixed_recovery'/'accepted.json',{'steps':[valid[s] for s in ids],'note':'Actual model rows; lossless wrapper normalization and validated sibling salvage.'})
        return valid

r.parse=parse
r.Client.rows=rows
if __name__=='__main__':r.main()
