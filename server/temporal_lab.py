"""Shared lab namespace for TEMPORAL research on Qwen2.5-Omni-3B (Thinker): does the video LM really understand time?
Default data: SVHalluc temporal tasks TA / TF / CMTB (983 questions).

The GPU server (server/kaggle_gpu_server.py) executes this file ONCE, loads the model, and then runs every job snippet in the
same namespace, so a job can directly use:
    MODEL_, PROC, L4                       model, processor, token ids of the letters A B C D
    fetch(name)                            file of the SVHalluc dataset (HF, pinned revision) -> local Path
    temporal_rows()                        all temporal rows of samples.json as (idx, row)
    clip_info(row)                         (path, duration_s, audio_codec)
    balanced_items(per_cell, seed, ...)    balanced per (task, gold letter), round robin, unusable clips skipped BEFORE selection
    parse_question(q) -> (stem, options)   prompt_of(stem, options, perm)   rotations(k)
    prepare_video(path)                    Qwen-official preprocessing (video via decord, audio via PyAV)
    qwen_inputs(PROC, prompt, media, mode) mode 'av' | 'vision' | 'audio';  move_inputs(inputs, MODEL_)
    score(media, stem, options, mode, fwd, both)
                                           log-prob of every option summed over all rotations of the option order;
                                           both=True also returns the log-probs in the ORIGINAL option order only
                                           (the protocol of published tables such as RED's), from the same forwards
    media_layout(MODEL_, PROC, inputs)     rope position ids, seconds per token, video / audio token masks (for interventions)
    narrated_block(audio)                  (start_s, end_s) of the narrated block of an SVHalluc clip (analysis only)
    run_loop(items, out_name, fn, show)    resumable loop (by idx) writing OUT/<out_name>, stops cleanly before the deadline
    OUT                                    results folder (download files with: python remote_client.py get <rel> <local>)

DATA FACTS (measured on the local copy, 2026-10-20):
  * temporal clips: 983 questions (TA 368, TF 276, CMTB 339), every clip has AAC audio (164/164 checked): no codec/label leak
    (unlike SVHalluc semantic, where gold-No clips are pcm and gold-Yes clips AAC).
  * the audio of a clip is digital silence + ONE continuous narrated block with whole-second edges (median ~13.6 s of a 23-28 s clip).
  * TA gold A/B balanced (184/184), TF A/B/C balanced (92 each), CMTB A/B/C balanced (113 each).
Labels are used only for scoring and analysis; nothing here feeds a label to the model."""
import os, sys, json, re, time, random
from pathlib import Path
import numpy as np
import torch

os.environ.setdefault('FORCE_QWENVL_VIDEO_READER', 'decord')
ROOT = Path(os.environ.get('LAB_ROOT', Path(globals().get('__file__', 'temporal_lab.py')).resolve().parent))
OUT = Path(os.environ.get('LAB_OUT', ROOT / 'out')); OUT.mkdir(parents=True, exist_ok=True)
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import av
from huggingface_hub import hf_hub_download

DATASET, REVISION = 'zcs15/SVHalluc', '211b4f14ab66ae9557b69e302d1f090e0e610839'
MODEL, MREV = 'Qwen/Qwen2.5-Omni-3B', 'f75b40e3da2003cdd6e1829b1f420ca70797c34e'
SR = 16000; BS = chr(92)
TEMPORAL_TASKS = ('TA', 'TF', 'CMTB')
NEUTRAL = 'What is happening in the video?'
MAX_DUR = float(os.environ.get('LAB_MAX_DUR', 42.0 if torch.cuda.device_count() >= 2 else 28.0))
DEADLINE = time.time() + float(os.environ.get('LAB_HOURS', 11.5)) * 3600


# ---------------------------------------------------------------- audio (PyAV only; no system ffmpeg is needed to read the clips)
def decode_audio(path):
    chunks = []
    with av.open(str(path)) as c:
        rs = av.AudioResampler(format='fltp', layout='mono', rate=SR)
        for fr in c.decode(audio=0):
            for p in rs.resample(fr): chunks.append(p.to_ndarray().reshape(-1))
        for p in rs.resample(None): chunks.append(p.to_ndarray().reshape(-1))
    return np.concatenate(chunks).astype('float32', copy=False)


def narrated_block(a):
    """SVHalluc audio = digital silence + one continuous narrated block: (start_s, end_s), or None. Analysis only."""
    hop = 800; n = len(a) // hop
    db = 20 * np.log10(np.sqrt((a[:n * hop].reshape(n, hop) ** 2).mean(1)) + 1e-9)
    on = np.where(db > -80)[0]
    return None if len(on) == 0 else (float(on[0] * 0.05), float((on[-1] + 1) * 0.05))


try:                                       # qwen-omni-utils reads the audio of an mp4 through librosa.load -> route it to PyAV
    import librosa
    _orig_load = librosa.load
    def _load(path, sr=None, offset=0.0, duration=None, **k):
        if str(path).lower().endswith('.mp4'):
            a = decode_audio(path)
            return a[int(sr * offset): None if duration is None else int(sr * (offset + duration))], sr
        return _orig_load(path, sr=sr, offset=offset, duration=duration, **k)
    librosa.load = _load
except ImportError:
    pass

from lab_native_video import prepare_video, qwen_inputs, move_inputs, SYSTEM
from speech_gate import media_layout


# ---------------------------------------------------------------- data
def fetch(name):
    local = os.environ.get('LAB_LOCAL_DATA')
    if local and (Path(local) / name).exists(): return Path(local) / name
    return Path(hf_hub_download(DATASET, name, repo_type='dataset', revision=REVISION))


def temporal_rows():
    rows = json.loads(fetch('samples.json').read_text(encoding='utf-8'))
    return [(i, r) for i, r in enumerate(rows) if r['data_type'] == 'temporal']


def clip_info(r):
    path = fetch(r['video_path'].replace(BS, '/'))
    with av.open(str(path)) as c:
        dur = float(c.duration / 1e6) if c.duration else None
        codec = c.streams.audio[0].codec_context.name if c.streams.audio else 'none'
    return path, dur, codec


def parse_question(q):
    parts = re.split(r'\s*\(([ABCD])\)\s*', q)
    return parts[0].strip(), [parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)]


SKIPPED = []


def usable(r, max_dur=None):
    try:
        _, dur, codec = clip_info(r)
        if codec == 'none': raise RuntimeError('no audio stream')
        if dur is None or dur > (max_dur or MAX_DUR): raise RuntimeError('long')
        return True
    except Exception as e:
        SKIPPED.append((r['task_type_abbr'], r['answer'], repr(e)[:60])); return False


def balanced_items(per_cell, seed, tasks=TEMPORAL_TASKS, max_dur=None, exclude=()):
    """per_cell items per (task, gold letter), shuffled with the seed, unusable clips skipped BEFORE selection, round robin."""
    cells = {}
    for i, r in temporal_rows():
        if r['task_type_abbr'] in tasks and i not in set(exclude): cells.setdefault((r['task_type_abbr'], r['answer']), []).append((i, r))
    pools = []
    for key in sorted(cells):
        pool = cells[key][:]; random.Random(f'{seed}-{key}').shuffle(pool); got = []
        for i, r in pool:
            if usable(r, max_dur): got.append((i, r))
            if len(got) == per_cell: break
        pools.append(got)
    out = []
    for k in range(max((len(p) for p in pools), default=0)):
        for p in pools:
            if k < len(p): out.append(p[k])
    return out


# ---------------------------------------------------------------- model and scoring
MODEL_ = PROC = L4 = None


def load_model():
    from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor
    cache = os.environ.get('LAB_CACHE')
    kw = dict(revision=MREV, torch_dtype=torch.float16, attn_implementation='sdpa', cache_dir=cache)
    if torch.cuda.device_count() >= 2: kw.update(device_map='balanced', max_memory={0: '10GiB', 1: '10GiB'})
    else: kw.update(device_map={'': 0})
    m = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(MODEL, **kw).eval()
    for p in m.parameters(): p.requires_grad_(False)
    return m, Qwen2_5OmniProcessor.from_pretrained(MODEL, revision=MREV, cache_dir=cache)


def rotations(k): return [tuple((s + j) % k for j in range(k)) for s in range(k)]
def prompt_of(stem, texts, perm): return stem + ' ' + ' '.join(f'({L}) {texts[j]}' for L, j in zip('ABCD', perm))


def last_logits(inp, mode='av'):
    """plain forward; lm_head only on the last position (saves memory / time)."""
    hk = MODEL_.lm_head.register_forward_pre_hook(lambda mod, args: (args[0][:, -1:, :],))
    try:
        with torch.no_grad(): return MODEL_(**inp, use_audio_in_video=(mode == 'av')).logits[0, -1].float()
    finally: hk.remove()


def score(media, stem, texts, mode='av', fwd=None, both=False):
    """log-prob of each option letter, summed over every rotation of the option order (removes letter-position bias).
    fwd(inputs) -> last-token logits replaces the plain forward (interventions).
    both=True returns (rotation_sum, canonical): canonical = log-probs with the options in their ORIGINAL order only
    (rotation 0), i.e. the protocol of tables that report plain accuracy; it comes from the same forwards."""
    k = len(texts); tot = np.zeros(k); canon = None
    for r_i, perm in enumerate(rotations(k)):
        inp, _ = qwen_inputs(PROC, prompt_of(stem, texts, perm), media, mode); inp = move_inputs(inp, MODEL_)
        lg = last_logits(inp, mode) if fwd is None else fwd(inp)
        lp = torch.log_softmax(lg[L4[:k]].cpu(), -1).numpy()
        if r_i == 0: canon = lp.round(3).tolist()                       # rotation 0 = identity order
        for p_, j in enumerate(perm): tot[j] += lp[p_]
        del inp
    torch.cuda.empty_cache()
    rs = tot.round(3).tolist()
    return (rs, canon) if both else rs


# ---------------------------------------------------------------- resumable loop
def time_left(): return DEADLINE - time.time()


def run_loop(items, out_name, fn, show=(), min_left=900):
    """fn(idx, row) -> dict.  One JSON line per item in OUT/out_name; items already in the file are skipped (resume).
    show: keys of score lists to print correct/wrong per (task, gold) as the run goes."""
    res = OUT / out_name
    done = {json.loads(l)['idx'] for l in res.read_text(encoding='utf-8').splitlines() if l.strip()} if res.exists() else set()
    print('items', len(items), 'already done', len(done), flush=True)
    stats, out = {}, []
    for n, (i, r) in enumerate(items):
        if i in done: continue
        if time_left() < min_left: print('deadline reached: stopping cleanly', flush=True); break
        t0 = time.time(); rec = dict(idx=i, task=r['task_type_abbr'], gold='ABCD'.index(r['answer']))
        try: rec.update(fn(i, r)); rec['sec'] = round(time.time() - t0)
        except torch.cuda.OutOfMemoryError: rec['error'] = 'OOM'
        except Exception as e: rec['error'] = repr(e)[:300]
        torch.cuda.empty_cache()
        with res.open('a', encoding='utf-8') as f: f.write(json.dumps(rec) + '\n')
        out.append(rec)
        if 'error' in rec: print(n + 1, len(items), rec['task'], i, 'ERR', rec['error'][:100], flush=True); continue
        g = rec['gold']; key = (rec['task'], 'ABC'[g])
        for c in show:
            if isinstance(rec.get(c), list): z = stats.setdefault((key, c), [0, 0]); z[int(np.argmax(rec[c])) != g] += 1
        marks = ' '.join(f'{c}={"Y" if int(np.argmax(rec[c])) == g else "-"}' for c in show if isinstance(rec.get(c), list))
        print(n + 1, len(items), rec['task'], i, marks, rec['sec'], 's', flush=True)
    if stats:
        print('correct/wrong per (task, gold):', {f'{k[0][0]}/{k[0][1]} {k[1]}': f'{v[0]}/{v[1]}' for k, v in sorted(stats.items())}, flush=True)
    return out
