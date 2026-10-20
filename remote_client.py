"""Client for the Kaggle GPU server. Config: remote_config.json {"url": ..., "token": ...} (next to this file) or env REMOTE_URL / REMOTE_TOKEN.
  python remote_client.py ping | info | restart
  python remote_client.py run job.py [--name N] [--nowait]     execute a Python file on the GPU server; prints its stdout and RESULT
  python remote_client.py status JOB_ID
  python remote_client.py upload LOCAL REMOTE_REL              -> OUT/uploads/REMOTE_REL
  python remote_client.py get REMOTE_REL LOCAL                 <- OUT/REMOTE_REL (e.g. rootcause_shard0of1.jsonl, hidden/h_1713.npz)"""
import os, sys, json, time, base64, urllib.request, urllib.error
from pathlib import Path

CFG = Path(__file__).with_name(os.environ.get('REMOTE_CFG', 'remote_config.json'))   # remote_config.json / remote_config2.json / remote_config3.json
cfg = json.loads(CFG.read_text(encoding='utf-8')) if CFG.exists() else {}
URL = (os.environ.get('REMOTE_URL') or cfg.get('url') or '').rstrip('/')
TOKEN = os.environ.get('REMOTE_TOKEN') or cfg.get('token') or ''


def call(method, path, body=None, raw=False, timeout=60):
    req = urllib.request.Request(URL + path, method=method, data=None if body is None else json.dumps(body).encode(),
                                 headers={'X-Token': TOKEN, 'Content-Type': 'application/json',
                                          'User-Agent': 'gpu-remote/1', 'ngrok-skip-browser-warning': '1'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read(); return data if raw else json.loads(data)
    except urllib.error.HTTPError as e:
        return {'error': f'HTTP {e.code}', 'detail': e.read().decode(errors='replace')[:300]}


def run_code(code, name='job', wait=True, poll=4.0):
    r = call('POST', '/submit', {'code': code, 'name': name})
    if 'job_id' not in r: return r
    jid = r['job_id']; print('job', jid, flush=True)
    if not wait: return r
    seen = 0
    while True:
        s = call('GET', f'/job/{jid}?since={seen}')
        if s.get('out'): print(s['out'], end='', flush=True)
        seen = s.get('out_len', seen)
        if s.get('status') in ('done', 'error'):
            if s.get('error'): print('\n--- ERROR ---\n' + s['error'])
            return s
        time.sleep(poll)


if __name__ == '__main__':
    a = sys.argv[1:]
    if not a: print(__doc__); sys.exit(0)
    if not URL: print('no URL: write remote_config.json or set REMOTE_URL'); sys.exit(1)
    cmd = a[0]
    if cmd == 'ping': print(json.dumps(call('GET', '/ping')))
    elif cmd == 'info': print(json.dumps(call('GET', '/info'), indent=1))
    elif cmd == 'restart': print(json.dumps(call('POST', '/restart', {})))
    elif cmd == 'status': print(json.dumps(call('GET', f'/job/{a[1]}'), indent=1))
    elif cmd == 'run':
        name = a[a.index('--name') + 1] if '--name' in a else Path(a[1]).stem
        s = run_code(Path(a[1]).read_text(encoding='utf-8'), name, wait='--nowait' not in a)
        if isinstance(s, dict) and 'result' in s: print('\nRESULT:', json.dumps(s['result'], indent=1, default=str)[:20000])
    elif cmd == 'upload': print(json.dumps(call('POST', '/upload', {'path': a[2], 'b64': base64.b64encode(Path(a[1]).read_bytes()).decode()})))
    elif cmd == 'get':
        d = call('GET', '/file?path=' + a[1], raw=True, timeout=300)
        if isinstance(d, bytes): Path(a[2]).write_bytes(d); print('saved', a[2], len(d), 'bytes')
        else: print(d)
    else: print(__doc__)
