"""Prefix a job file with parameter assignments, so one job file serves several servers / shards.
    python jobs/make_job.py jobs/baseline.py out.py PER_CELL=20 SHARD=0,2 OUT_NAME=baseline_s1.jsonl
    python remote_client.py run out.py --nowait
Values are Python literals; a comma list becomes a tuple (SHARD=0,2 -> (0, 2))."""
import ast, sys
from pathlib import Path

src, dst, *params = sys.argv[1:]
lines = []
for p in params:
    k, v = p.split('=', 1)
    try: val = ast.literal_eval(v)
    except (ValueError, SyntaxError): val = tuple(ast.literal_eval(x) for x in v.split(',')) if ',' in v else v
    if isinstance(val, str) and ',' in val: val = tuple(ast.literal_eval(x) for x in val.split(','))
    lines.append(f'{k} = {val!r}')
body = '\n'.join(lines) + '\n' + Path(src).read_text(encoding='utf-8')
ast.parse(body); Path(dst).write_text(body, encoding='utf-8')
print('wrote', dst, '|', '; '.join(lines))
