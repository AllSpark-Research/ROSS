"""Deterministic execution checks; no claim to establish mathematical truth."""
import json
import re


def parse_annotation(raw):
    try:
        # Decode JSON before interpreting its closing Markdown fence. Anchors
        # can legally contain literal ``` inside JSON strings.
        opening = re.search(r'(?m)^[ \t]*```(?:json)?[ \t]*\r?\n', raw)
        if opening:
            text = raw[opening.end():].lstrip()
            obj, end = json.JSONDecoder().raw_decode(text)
            if text[end:].strip() != '```':
                return None, 'revision_json_parse_error'
        else:
            obj = json.loads(raw.strip())
    except (ValueError, TypeError):
        return None, 'revision_json_parse_error'
    if not isinstance(obj, dict):
        return None, 'revision_schema_error'
    if any(type(obj.get(k)) is not bool for k in ('whole_clean', 'has_clean_final')):
        return None, 'revision_schema_error'
    spans = obj.get('keep_spans')
    if not isinstance(spans, list) or any(not isinstance(s, dict) or
            any(not isinstance(s.get(k), str) or not s[k] for k in
                ('start_anchor', 'end_anchor')) for s in spans):
        return None, 'revision_schema_error'
    if (obj['whole_clean'] and (not obj['has_clean_final'] or spans)) or (
            not obj['has_clean_final'] and spans) or (
            obj['has_clean_final'] and not obj['whole_clean'] and not spans):
        return None, 'revision_schema_error'
    return obj, None


def occurrences(text, quote):
    if not isinstance(quote, str) or not quote:
        return []
    return [(m.start(), m.start()+len(quote))
            for m in re.finditer('(?='+re.escape(quote)+')', text)]


def intersects(spans, interval):
    a,b = interval
    return any(x < b and a < y for x,y in spans)


def candidate_errors(response, parsed, resolved, answer_check, domain):
    """Validate exact anchors even if the legacy resolver said 'ok'."""
    errors=[]
    if not isinstance(parsed, dict) or not isinstance(resolved, dict):
        return ['missing_candidate']
    _, schema_error = parse_annotation(json.dumps(parsed))
    if schema_error:
        return [schema_error]
    spans=resolved.get('kept_spans', [])
    prev=0
    for s in spans:
        if not isinstance(s, (list,tuple)) or len(s)!=2 or any(type(x) is not int for x in s):
            return ['invalid_offsets']
        a,b=s
        if not 0<=a<b<=len(response) or a<prev:
            return ['invalid_offsets']
        prev=b
    if resolved.get('mode')=='dropped':
        return [] if not spans and parsed.get('has_clean_final') is False else ['inconsistent_drop']
    if not spans:
        errors.append('empty_supervision')
    if resolved.get('mode') not in ('whole_clean','spans_ok') or any(
            x!='ok' for x in resolved.get('spans_status',[])):
        errors.append('anchor_resolution_failure')
    if not parsed.get('whole_clean'):
        last_end=0
        for i,sp in enumerate(parsed.get('keep_spans',[])):
            starts=occurrences(response,sp.get('start_anchor'))
            ends=occurrences(response,sp.get('end_anchor'))
            if len(starts)!=1 or len(ends)!=1:
                errors.append(f'anchor_not_unique_or_missing:{i}')
                continue
            a,b=starts[0][0],ends[0][1]
            if a<last_end or b<=a:
                errors.append(f'anchor_order_invalid:{i}')
            last_end=b
    if spans and not answer_check(response,spans,domain):
        errors.append('final_answer_missing')
    return errors


def audit_evidence(response, spans, audit):
    """Exact full quotes only; ambiguous/unlocalized findings need review.

    checked_claims.fail must not silently disappear between audit and revision.
    Conservatively require the whole cited interval out of supervision. Never
    auto-delete it: quotes may contain context or the judge may be mistaken.
    """
    findings=[]; errors=[]
    if not isinstance(audit,dict) or audit.get('severity') not in ('none','minor','major') or any(
            type(audit.get(k)) is not bool for k in
            ('coherent','reaches_correct_answer','derivation_shown')):
        return [], ['audit_invalid']
    raw=audit.get('missed_masks',[])
    checks=audit.get('checked_claims',[])
    if not isinstance(raw,list) or not isinstance(checks,list):
        return [], ['audit_invalid']
    quotes=[(q,'missed_masks') for q in raw]
    for c in checks:
        if isinstance(c,dict) and c.get('verdict')=='fail':
            quotes.append((c.get('quote'),'checked_claims.fail'))
    for q,source in quotes:
        matches=occurrences(response,q)
        if len(matches)!=1:
            errors.append('audit_quote_ambiguous_or_not_verbatim')
            continue
        if intersects(spans,matches[0]):
            f={'quote':q,'interval':list(matches[0]),'source':source}
            if f not in findings:findings.append(f)
    if audit.get('severity')=='major' or audit.get('coherent') is False or any(
            audit.get(k) is False for k in ('reaches_correct_answer','derivation_shown')):
        errors.append('audit_requires_recheck')
    return findings, errors
