"""Build kaggle_temporal_gpu_server.ipynb: a Kaggle notebook that loads Qwen2.5-Omni-3B once and serves jobs over ngrok.
    python build_server_notebook.py
The notebook embeds every file of server/ (so the notebook alone is enough on Kaggle)."""
import ast, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILES = {n: (HERE / 'server' / n).read_text(encoding='utf-8') for n in ('temporal_lab.py', 'lab_native_video.py', 'speech_gate.py', 'kaggle_gpu_server.py')}
for s in FILES.values(): ast.parse(s)


def cell(kind, source):
    c = {'cell_type': kind, 'metadata': {}, 'source': source.splitlines(keepends=True)}
    if kind == 'code': ast.parse(source); c.update(execution_count=None, outputs=[])
    return c


intro = """# Server GPU cho nghiên cứu TEMPORAL (Video LM hiểu thời gian; SVHalluc TA / TF / CMTB): nạp Qwen2.5-Omni-3B một lần, nhận job qua ngrok

**Cách dùng** (chi tiết từng bước trong `README.md` của repo `Temporal_Hallucinate_VideoLLM`):
1. **Add-ons → Secrets → Add Secret**: Label `NGROK_AUTHTOKEN`, Value = authtoken ngrok của bạn, và **bật Attach** cho notebook này.
2. Session options: **GPU T4 ×2**, **Internet On**. Rồi **Save Version → Save & Run All (Commit)**.
3. Mở Logs của bản đang chạy. Sau khoảng 5–10 phút, log in ra **URL** (`https://….ngrok-free.dev`) và **TOKEN**. Ghi hai giá trị vào `remote_config.json` ở máy bạn (file này bị `.gitignore` chặn, **không bao giờ commit**).
4. Server sống tới hết phiên (tối đa 12 giờ). `python remote_client.py restart` khởi động lại server nếu job treo.

**Bảo mật.** Ai có cả URL lẫn TOKEN đều chạy được mã tuỳ ý trong notebook này. Không đăng log, không gửi URL/TOKEN vào kênh công khai. Đặt `ALLOW_EXEC = False` ở cell cuối nếu chỉ muốn thử kết nối. Dùng tài khoản Kaggle phụ; notebook cố ý không có vòng lặp chống ngắt."""

net_cell = '''# CELL 0 — kiểm tra Internet
import socket
for host in ('pypi.org', 'huggingface.co', 'ngrok.com'):
    try:
        socket.create_connection((host, 443), timeout=8).close(); print('OK  ', host)
    except OSError as e:
        raise RuntimeError('Không kết nối được ' + host + '. Bật Internet: Session options -> Internet = On.') from e
'''

setup_cell = '''# CELL 1 — môi trường: venv riêng với các phiên bản đã ghim (transformers 4.57.6 cho Qwen2.5-Omni), giữ nguyên torch của Kaggle
import os, sys, json, subprocess, importlib.metadata, venv, time
from pathlib import Path
assert Path('/kaggle').exists(), 'Chạy trên Kaggle (GPU T4 x2, Internet On).'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'; os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True'); os.environ['FORCE_QWENVL_VIDEO_READER'] = 'decord'
ROOT = Path('/kaggle/working/temporal_lab_runner'); ROOT.mkdir(parents=True, exist_ok=True)
OUT = Path('/kaggle/working/temporal_lab_' + time.strftime('%Y%m%d_%H%M%S')); OUT.mkdir(parents=True, exist_ok=True)
ENV = ROOT / 'env'
venv.EnvBuilder(system_site_packages=True, with_pip=False).create(ENV)          # no ensurepip; the host pip installs into it
PYTHON = ENV / 'bin' / 'python'
PIP = [sys.executable, '-m', 'pip', '--python', str(PYTHON)]
keep = []
for p in ('torch', 'torchvision', 'torchaudio'):
    try: keep.append(p + '==' + importlib.metadata.version(p))
    except importlib.metadata.PackageNotFoundError: pass
(ROOT / 'keep_torch.txt').write_text(chr(10).join(keep) + chr(10), encoding='utf-8')
packages = ['transformers==4.57.6', 'accelerate==1.12.0', 'huggingface-hub==0.36.0', 'qwen-omni-utils==0.0.8', 'decord==0.6.0', 'av==16.0.1',
            'librosa==0.11.0', 'soundfile>=0.12,<0.14', 'numpy>=2,<2.4', 'Pillow>=10,<13', 'sentencepiece>=0.2,<0.3', 'protobuf>=5,<7']
with (OUT / 'install.log').open('a', encoding='utf-8') as log:
    proc = subprocess.Popen([*PIP, 'install', '--only-binary=:all:', '--constraint', str(ROOT / 'keep_torch.txt'), *packages],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
    for line in proc.stdout: print(line, end='', flush=True); log.write(line)
    if proc.wait(): raise RuntimeError('Cài thư viện thất bại; xem ' + str(OUT / 'install.log'))
subprocess.run([str(PYTHON), '-c', 'import torch, av, decord; from transformers import Qwen2_5OmniThinkerForConditionalGeneration; '
                'print("torch", torch.__version__, "| GPUs", torch.cuda.device_count(), "| av", av.__version__)'], check=True)
print('Môi trường sẵn sàng. Kết quả và log:', OUT)
'''

write_cell = '# CELL 2 — ghi mã nguồn server\nSERVER_FILES = ' + repr(FILES) + '\n' + \
    "for _n, _s in SERVER_FILES.items(): (ROOT / _n).write_text(_s, encoding='utf-8')\nprint('Đã ghi:', list(SERVER_FILES))\n"

serve_cell = '''# CELL 3 — khởi động server + ngrok, giữ notebook sống
ALLOW_EXEC = True      # False = chỉ kiểm tra kết nối, server không nhận mã
PORT = 8765
import secrets, subprocess, time, sys, os, urllib.request
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'pyngrok'], check=True)
from pyngrok import ngrok

def ngrok_token():
    t = os.environ.get('NGROK_AUTHTOKEN', '')
    if t: return t
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret('NGROK_AUTHTOKEN')
    except Exception as e:
        raise RuntimeError('Không đọc được Kaggle Secret NGROK_AUTHTOKEN (' + type(e).__name__ + '). Add-ons -> Secrets -> thêm NGROK_AUTHTOKEN và bật Attach.')

TOKEN = secrets.token_urlsafe(32)
env = {**os.environ, 'GPU_SERVER_TOKEN': TOKEN, 'GPU_SERVER_ALLOW_EXEC': '1' if ALLOW_EXEC else '0', 'PYTHONUNBUFFERED': '1'}
cmd = [str(PYTHON), str(ROOT / 'kaggle_gpu_server.py'), str(ROOT), str(OUT), str(PORT)]
SERVER_LOG = OUT / 'server.log'

def start_server():
    return subprocess.Popen(cmd, stdout=open(SERVER_LOG, 'a', encoding='utf-8'), stderr=subprocess.STDOUT, env=env)

def wait_ready(timeout=1200):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(f'http://127.0.0.1:{PORT}/ping', timeout=3); return True
        except Exception:
            time.sleep(5)
    return False

server = start_server()
print('Đang nạp model (3–6 phút lần đầu)...', flush=True)
if not wait_ready():
    print(SERVER_LOG.read_text()[-3000:]); raise RuntimeError('server không lên được; xem server.log ở trên')
ngrok.set_auth_token(ngrok_token())
public_url = ngrok.connect(PORT, 'http')
print('=' * 70)
print('  URL   :', public_url)
print('  TOKEN :', TOKEN)
print('=' * 70, flush=True)

while True:                       # giữ notebook sống; tự khởi động lại server khi /restart hoặc khi nó chết
    time.sleep(30)
    if server.poll() is not None:
        print('server dừng (mã', server.returncode, ') -> khởi động lại', flush=True)
        server = start_server()
        if not wait_ready(): print('khởi động lại thất bại; xem server.log', flush=True)
'''

nb = {'cells': [cell('markdown', intro), cell('code', net_cell), cell('code', setup_cell), cell('code', write_cell), cell('code', serve_cell)],
      'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python', 'version': '3.13'}},
      'nbformat': 4, 'nbformat_minor': 5}
target = HERE / 'kaggle_temporal_gpu_server.ipynb'
target.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
print(target.name, target.stat().st_size, 'bytes')
