"""Persistent GPU job server (runs inside the Kaggle venv; reached through an ngrok tunnel).
Usage: GPU_SERVER_TOKEN=... [GPU_SERVER_ALLOW_EXEC=1] python kaggle_gpu_server.py ROOT OUT PORT

The model (Qwen2.5-Omni-3B Thinker, fp16) is loaded ONCE into a namespace together with every helper of temporal_lab.py
(fetch, balanced_items, prepare_video, qwen_inputs, score, media_layout, run_loop, ...). Jobs are Python snippets executed in
that namespace by one worker thread (the GPU is single-tenant). A job may set RESULT (JSON-serialisable) and print.

Endpoints (all but /ping need the header X-Token):
  GET  /ping                         -> {"ok": true}
  GET  /info                         -> gpu / memory / queue state
  POST /submit  {code, name}         -> {job_id}           (needs ALLOW_EXEC)
  GET  /job/<id>?since=<n>           -> status, new stdout lines, result/error
  POST /upload  {path, b64}          -> writes OUT/uploads/<path>
  GET  /file?path=<rel>              -> bytes of OUT/<rel>
  POST /restart                      -> the process exits; the notebook restarts it (frees a hung GPU)
The server only listens on 127.0.0.1; the tunnel forwards to it."""
import os, sys, json, time, threading, traceback, io, contextlib, base64, hmac, uuid, queue
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

ROOT, OUT, PORT = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
TOKEN = os.environ['GPU_SERVER_TOKEN']
ALLOW_EXEC = os.environ.get('GPU_SERVER_ALLOW_EXEC', '1') == '1'
UP = OUT / 'uploads'; UP.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(UP)); sys.path.insert(0, str(ROOT))
os.environ['LAB_ROOT'] = str(ROOT); os.environ['LAB_OUT'] = str(OUT)

# ---- namespace with the model and all helpers ------------------------------------------------------------------------
NS = {'__name__': 'labns', '__file__': str(ROOT / 'temporal_lab.py')}
exec(compile((ROOT / 'temporal_lab.py').read_text(encoding='utf-8'), 'temporal_lab.py', 'exec'), NS)
t0 = time.time()
NS['MODEL_'], NS['PROC'] = NS['load_model']()
NS['L4'] = [NS['PROC'].tokenizer.convert_tokens_to_ids(L) for L in 'ABCD']
print(f'model loaded in {time.time() - t0:.0f}s', flush=True)

JOBS, Q, LOCK = {}, queue.Queue(), threading.Lock()


class Tee(io.TextIOBase):
    def __init__(self, job): self.job = job
    def write(self, s):
        with LOCK: self.job['out'] += s
        sys.__stdout__.write(s); return len(s)
    def flush(self): sys.__stdout__.flush()


def worker():
    import torch
    while True:
        jid = Q.get(); job = JOBS[jid]; job['status'] = 'running'; job['t_start'] = time.time()
        NS['RESULT'] = None
        try:
            with contextlib.redirect_stdout(Tee(job)):
                exec(compile(job['code'], f'<job {job["name"]}>', 'exec'), NS)
            blob = json.dumps(NS.get('RESULT'), default=str)
            job['result'] = json.loads(blob) if len(blob) < 5_000_000 else {'_too_large': len(blob), 'hint': 'write it to a file and GET /file'}
            job['status'] = 'done'
        except BaseException:
            job['error'] = traceback.format_exc(); job['status'] = 'error'
        finally:
            try: torch.cuda.empty_cache()
            except Exception: pass
            job['t_end'] = time.time()


threading.Thread(target=worker, daemon=True).start()


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def _send(self, code, obj=None, raw=None, ctype='application/json'):
        body = raw if raw is not None else json.dumps(obj, default=str).encode()
        self.send_response(code); self.send_header('Content-Type', ctype); self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)

    def _auth(self): return hmac.compare_digest(self.headers.get('X-Token', ''), TOKEN)

    def _body(self):
        n = int(self.headers.get('Content-Length', 0)); return json.loads(self.rfile.read(n) or b'{}')

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == '/ping': return self._send(200, {'ok': True})
        if not self._auth(): return self._send(401, {'error': 'bad token'})
        if u.path == '/info':
            import torch
            return self._send(200, {'gpus': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                                    'mem_alloc_GiB': [round(torch.cuda.memory_allocated(i) / 2 ** 30, 2) for i in range(torch.cuda.device_count())],
                                    'queue': Q.qsize(), 'jobs': len(JOBS), 'allow_exec': ALLOW_EXEC, 'uptime_s': round(time.time() - t0)})
        if u.path.startswith('/job/'):
            jid = u.path.split('/')[2]; since = int(parse_qs(u.query).get('since', ['0'])[0]); j = JOBS.get(jid)
            if not j: return self._send(404, {'error': 'no such job'})
            with LOCK: out = j['out']
            return self._send(200, {'status': j['status'], 'name': j['name'], 'out': out[since:], 'out_len': len(out), 'result': j.get('result'),
                                    'error': j.get('error'), 'seconds': round((j.get('t_end') or time.time()) - j.get('t_start', time.time()), 1) if j.get('t_start') else 0})
        if u.path == '/file':
            rel = parse_qs(u.query).get('path', [''])[0]; p = (OUT / rel).resolve()
            if OUT.resolve() not in p.parents or not p.is_file(): return self._send(404, {'error': 'not found'})
            return self._send(200, raw=p.read_bytes(), ctype='application/octet-stream')
        return self._send(404, {'error': 'unknown'})

    def do_POST(self):
        u = urlparse(self.path)
        if not self._auth(): return self._send(401, {'error': 'bad token'})
        b = self._body()
        if u.path == '/submit':
            if not ALLOW_EXEC: return self._send(403, {'error': 'exec disabled on this server'})
            jid = uuid.uuid4().hex[:10]; JOBS[jid] = {'name': b.get('name', 'job'), 'code': b['code'], 'status': 'queued', 'out': ''}; Q.put(jid)
            return self._send(200, {'job_id': jid})
        if u.path == '/upload':
            rel = Path(b['path']); p = (UP / rel).resolve()
            if UP.resolve() not in p.parents: return self._send(400, {'error': 'path must stay under uploads/'})
            p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(base64.b64decode(b['b64'])); return self._send(200, {'written': str(rel), 'bytes': p.stat().st_size})
        if u.path == '/restart':
            self._send(200, {'restarting': True}); threading.Thread(target=lambda: (time.sleep(0.5), os._exit(3)), daemon=True).start(); return
        return self._send(404, {'error': 'unknown'})


print(f'serving on 127.0.0.1:{PORT} (allow_exec={ALLOW_EXEC})', flush=True)
ThreadingHTTPServer(('127.0.0.1', PORT), H).serve_forever()
