"""Analysis of jobs/baseline.py.   python analysis/analyze_baseline.py s1.jsonl [s2.jsonl ...]
Per (task, gold): correct/wrong and prediction histogram for each condition, for BOTH scoring protocols:
  rot   log-probs summed over all rotations of the option order (position-bias free)
  canon options in their original order only (plain accuracy, as in published tables)
Also: plain accuracy per task and balanced accuracy per task (mean over gold cells) for each condition."""
import collections, json, sys
import numpy as np

R = [json.loads(l) for f in sys.argv[1:] for l in open(f, encoding='utf-8') if l.strip()]
err = collections.Counter(r['error'][:50] for r in R if 'error' in r); R = [r for r in R if 'error' not in r]
seen = set(); R = [r for r in R if not (r['idx'] in seen or seen.add(r['idx']))]           # duplicates (e.g. a smoke file) removed
am = lambda v: int(np.argmax(v)); L = 'ABC'
CONDS = [c for c in ('av', 'vision', 'audio') if any(c in r for r in R)]
TASKS = [t for t in ('TA', 'TF', 'CMTB') if any(r['task'] == t for r in R)]
print(f'items {len(R)} | errors {dict(err)} | per task {dict(collections.Counter(r["task"] for r in R))}')
cells = [(t, g) for t in TASKS for g in sorted({r['gold'] for r in R if r['task'] == t})]
for proto, suffix in (('rot (all rotations)', ''), ('canon (original option order)', '_canon')):
    print(f'\n===== {proto}')
    print('  condition  ' + ' '.join(f'{t}/{L[g]:<7s}' for t, g in cells) + ' ' + ' '.join(f'plain-{t:<5s}' for t in TASKS) + ' ' + ' '.join(f'bal-{t:<5s}' for t in TASKS))
    for c in CONDS:
        key = c + suffix; row, plain, bal = [], {}, {}
        for t, g in cells:
            S = [r for r in R if r['task'] == t and r['gold'] == g and key in r]; k = sum(am(r[key]) == g for r in S); row.append(f'{k}/{len(S) - k}')
        for t in TASKS:
            S = [r for r in R if r['task'] == t and key in r]
            plain[t] = 100 * np.mean([am(r[key]) == r['gold'] for r in S]) if S else float('nan')
            bal[t] = 100 * np.mean([np.mean([am(r[key]) == g for r in S if r['gold'] == g]) for tt, g in cells if tt == t])
        print(f'  {c:9s}  ' + ' '.join(f'{x:<9s}' for x in row) + ' ' + ' '.join(f'{plain[t]:10.1f}' for t in TASKS) + ' ' + ' '.join(f'{bal[t]:8.1f}' for t in TASKS))
    print('  predictions (A/B[/C]) per task:')
    for c in CONDS:
        key = c + suffix
        print(f'  {c:9s}  ' + '   '.join(f'{t}: ' + '/'.join(str(sum(r["task"] == t and am(r[key]) == j for r in R)) for j in range(max(r['n_opts'] for r in R if r['task'] == t))) for t in TASKS))
print('\nChance: TA 50%, TF 33%, CMTB 33%.')
