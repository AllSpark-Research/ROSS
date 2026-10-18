"""Validate step-native agentic exports and construct annotation windows."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import time
import urllib.request

HERE = Path(__file__).resolve().parent
CALL = re.compile(r'<tool_call>\s*<function=([^>]+)>(.*?)</function>\s*</tool_call>', re.S)
PARAM = re.compile(r'<parameter=([^>]+)>(.*?)</parameter>', re.S)
RETURN = re.compile(r'<tool_response>(.*?)</tool_response>', re.S)
PROTOCOL = re.compile(r'<\|[^>]+\|>|</?think>|</?tool_call>|</?function(?:=[^>]+)?>|</?parameter(?:=[^>]+)?>')

def read(p):
    return json.loads(Path(p).read_text())

def dump(p, x):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(x, ensure_ascii=False, indent=2))
    tmp.replace(p)

def digest(x):
    return hashlib.sha256(json.dumps(x, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

def parse_content(text):
    # Accept exactly one complete JSON code block, including a prose preface.
    # Never repair JSON, quotes or source spans; reject ambiguous multiple blocks.
    text = text.strip()
    if text.startswith('{'):
        return json.loads(text)
    fences = re.findall(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.S)
    if len(fences) != 1:
        raise ValueError('One complete JSON object/code block required')
    return json.loads(fences[0])

def build(data):
    """Regex extracts boundaries, never semantic labels. Fail on ambiguity."""
    sources = {'TASK': data['initial_context_raw']}
    steps = []
    seen = set()
    for original in data['steps']:
        sid, text = original['id'], original['assistant_raw']
        if sid in sources:
            raise ValueError('Duplicate source ID: ' + sid)
        matches = list(CALL.finditer(text))
        if len(matches) != len(original['calls']) or text.count('<tool_call>') != len(matches):
            raise ValueError('Ambiguous tool serialization; needs structured-message alignment: ' + sid)
        parts, payloads = [], []
        previous = 0
        for match, meta in zip(matches, original['calls']):
            if match.group(1) != meta['tool_name'] or meta['tool_call_id'] in seen:
                raise ValueError('Tool name or unique ID mismatch')
            seen.add(meta['tool_call_id'])
            if not original['shared_thought_and_text'] == text[:matches[0].start()]:
                raise ValueError('Shared thought boundary mismatch')
            if text[previous:match.start()].strip():
                parts.append({'kind': 'thought_or_text', 'start': previous, 'end': match.start()})
            params = list(PARAM.finditer(match.group(2)))
            if (len(params) != match.group(2).count('<parameter=')
                    or PARAM.sub('', match.group(2)).strip()
                    or len({m.group(1) for m in params}) != len(params)):
                raise ValueError('Ambiguous parameter serialization')
            arguments = []
            for parameter in params:
                a, b = parameter.span(2)
                a += match.start(2)
                b += match.start(2)
                payloads.append((a, b))
                arguments.append({'name': parameter.group(1), 'start': a, 'end': b})
            parts.append({'kind': 'tool_call', 'start': match.start(), 'end': match.end(),
                          'tool_name': meta['tool_name'], 'tool_call_id': meta['tool_call_id'],
                          'arguments': arguments})
            previous = match.end()
        if previous < len(text):
            parts.append({'kind': 'thought_or_text', 'start': previous, 'end': len(text)})
        if ''.join(text[p['start']:p['end']] for p in parts) != text:
            # Whitespace gaps are still original data; include them instead of dropping.
            covered = sorted((p['start'], p['end']) for p in parts)
            cursor = 0
            for a, b in covered:
                if text[cursor:a].strip():
                    raise ValueError('Uncovered assistant content')
                cursor = b
        protected = [m.span() for m in PROTOCOL.finditer(text)
                     if not any(a <= m.start() and m.end() <= b for a, b in payloads)]
        obs = original['observation_raw']
        returns = RETURN.findall(obs or '')
        if len(returns) != sum(c['raw_return'] is not None for c in original['calls']):
            raise ValueError('Return count mismatch')
        if returns != [c['raw_return'] for c in original['calls'] if c['raw_return'] is not None]:
            raise ValueError('Return order/text mismatch')
        for c in original['calls']:
            if c['raw_return'] is not None and not c.get('return_original_text_verified'):
                raise ValueError('Unverified return link')
        sources[sid] = text
        if obs is not None:
            sources['OBS_' + sid] = obs
        steps.append({'id': sid, 'parts': parts, 'protected': protected,
                      'assistant_tokens': original['assistant_tokens'],
                      'calls': [{'id': c['tool_call_id'], 'name': c['tool_name'],
                                 'return_available': c['raw_return'] is not None}
                                for c in original['calls']]})
    return {'sample': data['summary']['sample'], 'sources': sources, 'steps': steps}

def windows(state, max_steps=4, max_chars=12000):
    result, current, size = [], [], 0
    for step in state['steps']:
        n = len(state['sources'][step['id']])
        if current and (len(current) >= max_steps or size + n > max_chars):
            result.append(current)
            current, size = [], 0
        current.append(step['id'])
        size += n
    if current:
        result.append(current)
    return result

def packet(state, owned, review=None):
    # Original order/full evidence retained. Repeat only owned-neighbor evidence at
    # the tail to reduce confusing an old snippet with the latest observation.
    step_ids = [s['id'] for s in state['steps']]
    focus = {}
    for sid in owned:
        index = step_ids.index(sid)
        for key in ([('OBS_' + step_ids[index - 1]) if index else 'TASK', sid, 'OBS_' + sid]):
            if key in state['sources']:
                focus[key] = state['sources'][key]
    result = {'original_sources_in_order': [{'source_id': k, 'text': v} for k, v in state['sources'].items()],
              'step_directory': [{k: v for k, v in s.items() if k != 'protected'} for s in state['steps']],
              'owned_step_ids': owned, 'focused_original_sources': focus,
              'instruction': 'Judge only owned_step_ids. Full original history is context, not extra output scope.'}
    if review is not None:
        result['fallible_first_pass'] = review
        result['instruction'] += (' Independently reread ALL owned thought and action text for missed defects and over-masking. '
                                  'Return full replacement decisions, not a diff. First-pass labels are not evidence.')
    return json.dumps(result, ensure_ascii=False)

def positions(text, quote):
    return [m.start() for m in re.finditer('(?=' + re.escape(quote) + ')', text)]

def locate(text, span):
    quote, occurrence = span.get('quote'), span.get('occurrence')
    if not isinstance(quote, str) or not quote or type(occurrence) is not int or occurrence < 0:
        raise ValueError('Exact quote and nonnegative integer occurrence required')
    hits = positions(text, quote)
    if occurrence >= len(hits) or not span.get('reason'):
        raise ValueError('Quote missing/occurrence invalid/reason absent')
    return hits[occurrence], hits[occurrence] + len(quote)

def overlap(a, b):
    return max(a[0], b[0]) < min(a[1], b[1])

def validate(obj, state, owned):
    decisions = obj.get('steps')
    if not isinstance(decisions, list) or [d.get('id') for d in decisions] != owned:
        raise ValueError('Exactly one decision per owned step, in order, required')
    lookup = {s['id']: s for s in state['steps']}
    source_order = {sid: i for i, sid in enumerate(state['sources'])}
    def check_evidence(ev):
        source, quote = ev.get('source_id'), ev.get('quote')
        if source not in state['sources'] or not isinstance(quote, str) or not quote or quote not in state['sources'][source]:
            raise ValueError('Evidence quote absent from its exact source')
        return source
    resolved = []
    for decision in decisions:
        sid = decision['id']
        text = state['sources'][sid]
        excludes, unknown = decision.get('exclude'), decision.get('unresolved')
        if not isinstance(excludes, list) or not isinstance(unknown, list) or not decision.get('summary'):
            raise ValueError('Summary and explicit arrays required')
        expected = 'uncertain' if unknown else 'mixed' if excludes else 'clean'
        if decision.get('status') != expected:
            raise ValueError('Status must agree with span arrays')
        action = decision.get('action_time', {})
        if action.get('assessment') not in {'reasonable', 'defective', 'mixed', 'unknown'} or not action.get('reason'):
            raise ValueError('Action-time assessment required')
        if not isinstance(action.get('evidence'), list) or not action['evidence']:
            raise ValueError('Action-time evidence required')
        for ev in action['evidence']:
            if source_order[check_evidence(ev)] > source_order[sid]:
                raise ValueError('Later evidence cannot be action-time knowledge')
        impact = decision.get('impact', {})
        if impact.get('assessment') not in {'useful', 'harmful', 'mixed', 'no_demonstrated_effect', 'unknown'} or not impact.get('reason') or not isinstance(impact.get('links'), list):
            raise ValueError('Downstream impact assessment and links required')
        if impact['assessment'] in {'useful', 'harmful', 'mixed'} and not impact['links']:
            raise ValueError('Demonstrated impact needs downstream evidence')
        for link in impact['links']:
            if source_order[check_evidence(link)] <= source_order[sid]:
                raise ValueError('Impact must cite a later observation/step')
            if link.get('relation') not in {'informs', 'enables_repair', 'propagates_error', 'changes_state', 'contradicts'} or link.get('basis') not in {'observed', 'inferred'} or not link.get('reason'):
                raise ValueError('Explicit impact relation, basis and reason required')
        step_mask = decision.get('step_mask')
        if step_mask not in {'retain', 'selective', 'exclude_all'} or (step_mask == 'retain') != (not excludes):
            raise ValueError('Step mask summary must agree with explicit excludes')
        selected, preserved, unresolved = [], [], []
        for span in excludes:
            boundary = locate(text, span)
            evidence = span.get('evidence')
            if not isinstance(evidence, list) or not evidence:
                raise ValueError('Every exclusion requires exact evidence')
            for ev in evidence:
                check_evidence(ev)
            keep = span.get('preserve')
            if not isinstance(keep, list):
                raise ValueError('preserve array required')
            preserved.extend(locate(text, q) for q in keep)
            selected.append(boundary)
            resolved.append({'region_id': sid, 'char_start': boundary[0], 'char_end': boundary[1],
                             'resolved_quote': span['quote'], 'rationale': span['reason'], 'evidence': evidence})
        for span in unknown:
            unresolved.append(locate(text, span))
            if not span.get('need'):
                raise ValueError('Unresolved span requires specific missing evidence')
        if any(overlap(a, b) for i, a in enumerate(selected) for b in selected[i + 1:]):
            raise ValueError('Overlapping/duplicate excludes')
        if any(overlap(a, b) for a in selected for b in preserved + unresolved + lookup[sid]['protected']):
            raise ValueError('Exclusion overlaps preserved, unresolved or protocol text')
        if step_mask == 'exclude_all':
            covered = selected + lookup[sid]['protected']
            if any(not char.isspace() and not any(a <= i < b for a, b in covered) for i, char in enumerate(text)):
                raise ValueError('exclude_all requires complete non-protocol content coverage')
    return resolved

def request(state, owned, dest, args, review=None):
    body = {'model': args.model, 'messages': [{'role': 'system', 'content': (HERE / 'prompt.txt').read_text()},
                                           {'role': 'user', 'content': packet(state, owned, review)}],
            'reasoning_effort': 'high', 'chat_template_kwargs': {'enable_thinking': True},
            'temperature': 0, 'max_tokens': args.max_tokens, 'stream': False}
    fingerprint = digest({'endpoint': args.endpoint, 'body': body})
    accepted = dest / 'accepted.json'
    if accepted.exists():
        cache = read(accepted)
        if cache['input_sha256'] != fingerprint:
            raise ValueError('Cached request changed; use a new output directory')
        saved_response = read(dest / cache['response_file'])
        if digest(saved_response) != cache['response_sha256'] or parse_content(saved_response['choices'][0]['message']['content']) != cache['result']:
            raise ValueError('Cached response integrity mismatch')
        validate(cache['result'], state, owned)
        return cache['result']
    prefix = len(list(dest.glob('*_request.json')))
    for attempt in range(2):
        path = dest / str(prefix + attempt)
        dump(path.with_name(path.name + '_request.json'), body)
        start = time.monotonic()
        headers = {'Content-Type': 'application/json'}
        key = os.environ.get('ROSS_API_KEY', '')
        if key:
            name = os.environ.get('ROSS_AUTH_HEADER', 'Authorization')
            scheme = os.environ.get('ROSS_AUTH_SCHEME', 'Bearer')
            headers[name] = f'{scheme} {key}'.strip()
        # Transport/service errors propagate; do not disguise them as labeling errors.
        req = urllib.request.Request(args.endpoint, data=json.dumps(body).encode(), headers=headers)
        with urllib.request.urlopen(req, timeout=600) as response:
            raw = json.load(response)
        dump(path.with_name(path.name + '_response.json'), raw)
        try:
            choice = raw['choices'][0]
            if choice['finish_reason'] != 'stop':
                raise ValueError('Incomplete response: ' + str(choice['finish_reason']))
            result = parse_content(choice['message']['content'])
            validate(result, state, owned)
            dump(accepted, {'input_sha256': fingerprint, 'result': result, 'elapsed_seconds': time.monotonic() - start,
                            'response_file': path.name + '_response.json', 'response_sha256': digest(raw)})
            return result
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            dump(path.with_name(path.name + '_error.json'), {'error': str(exc)})
            body['messages'] = body['messages'][:2] + [{'role': 'user', 'content':
                'Your response failed validation: ' + str(exc) + '. Reread the exact original data and return the complete JSON.'}]
    raise ValueError('Two invalid attempts; this window remains unannotated')

def prepare(inputs, output, max_steps, max_chars):
    summaries = []
    for path in sorted(Path(inputs).glob('*.steps.json')):
        state = build(read(path))
        groups = windows(state, max_steps, max_chars)
        dest = Path(output) / state['sample']
        dump(dest / 'state.json', state)
        dump(dest / 'windows.json', groups)
        for i, group in enumerate(groups):
            dump(dest / 'packets' / f'{i:03d}.json', json.loads(packet(state, group)))
        summaries.append({'sample': state['sample'], 'steps': len(state['steps']),
                          'primary_windows': len(groups), 'window_step_counts': [len(g) for g in groups],
                          'oversized_single_steps': [g[0] for g in groups if len(g) == 1 and len(state['sources'][g[0]]) > max_chars]})
    if not summaries:
        raise ValueError('No step exports found')
    dump(Path(output) / 'plan.json', {'max_steps': max_steps, 'max_owned_chars': max_chars, 'samples': summaries,
                                      'note': 'No API calls. Character limit covers owned assistant text, not full request/context.'})
    return summaries

def run(args):
    root = args.output
    config = {'endpoint': args.endpoint, 'model': args.model, 'max_tokens': args.max_tokens,
              'review': args.review, 'prompt_sha256': digest((HERE / 'prompt.txt').read_text()),
              'code_sha256': digest(Path(__file__).read_text()),
              'inputs': {str(p.relative_to(args.prepared)): digest(read(p))
                         for p in sorted(args.prepared.glob('*/state.json')) + sorted(args.prepared.glob('*/windows.json'))}}
    if (root / 'config.json').exists() and read(root / 'config.json') != config:
        raise ValueError('Configuration/input changed; use a new output directory')
    dump(root / 'config.json', config)
    jobs, states = [], {}
    for path in sorted(args.prepared.glob('*/state.json')):
        state = read(path)
        states[state['sample']] = state
        for i, group in enumerate(read(path.with_name('windows.json'))):
            jobs.append((state, group, root / state['sample'] / 'primary' / f'{i:03d}'))
    if not jobs:
        raise ValueError('No prepared windows')
    decisions = {sample: {} for sample in states}
    errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(request, s, group, dest, args): (s, group) for s, group, dest in jobs}
        for future in concurrent.futures.as_completed(futures):
            state, group = futures[future]
            try:
                decisions[state['sample']].update((d['id'], d) for d in future.result()['steps'])
            except Exception as exc:
                errors.append({'phase': 'primary', 'sample': state['sample'], 'steps': group, 'error': str(exc)})
            dump(root / 'progress.json', {'completed_steps': sum(len(d) for d in decisions.values()),
                                         'total_steps': sum(len(s['steps']) for s in states.values()), 'errors': errors})
    for sample, state in states.items():
        ids = [s['id'] for s in state['steps']]
        if set(decisions[sample]) != set(ids):
            continue
        first = {'steps': [decisions[sample][sid] for sid in ids]}
        dump(root / sample / 'first_pass.json', first)
        if args.review:
            selected = [sid for sid in ids if decisions[sample][sid]['status'] != 'clean'
                        or not next(s for s in state['steps'] if s['id'] == sid)['calls']]
            for i in range(0, len(selected), 4):
                group = selected[i:i + 4]
                try:
                    reviewed = request(state, group, root / sample / 'review' / f'{i // 4:03d}', args,
                                       {'steps': [decisions[sample][sid] for sid in group]})
                    decisions[sample].update((d['id'], d) for d in reviewed['steps'])
                except Exception as exc:
                    errors.append({'phase': 'review', 'sample': sample, 'steps': group, 'error': str(exc)})
        result = {'steps': [decisions[sample][sid] for sid in ids]}
        spans = validate(result, state, ids)
        dump(root / sample / 'annotation.json', {'decisions': result['steps'], 'mask_spans': spans,
             'eligible_for_sft': False, 'note': 'Character-span intermediate; token compilation and final semantic validation are separate steps.',
             'review_complete': args.review and not any(e['sample'] == sample and e['phase'] == 'review' for e in errors)})
    dump(root / 'STATUS.json', {'complete': not errors, 'errors': errors, 'eligible_for_sft': False})
    if errors:
        raise SystemExit(1)

def main():
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers(dest='command', required=True)
    prep = subs.add_parser('prepare')
    prep.add_argument('--inputs', type=Path, required=True)
    prep.add_argument('--output', type=Path, required=True)
    prep.add_argument('--max-steps', type=int, default=4)
    prep.add_argument('--max-chars', type=int, default=12000)
    remote = subs.add_parser('run')
    remote.add_argument('--prepared', type=Path, required=True)
    remote.add_argument('--output', type=Path, required=True)
    remote.add_argument('--endpoint', required=True)
    remote.add_argument('--model', default='judge-model')
    remote.add_argument('--max-tokens', type=int, default=32768)
    remote.add_argument('--workers', type=int, default=4)
    remote.add_argument('--review', action='store_true')
    args = parser.parse_args()
    if args.command == 'prepare':
        if args.max_steps < 1 or args.max_chars < 1:
            parser.error('Limits must be positive')
        print(json.dumps(prepare(args.inputs, args.output, args.max_steps, args.max_chars), ensure_ascii=False, indent=2))
    else:
        run(args)

if __name__ == '__main__':
    main()
