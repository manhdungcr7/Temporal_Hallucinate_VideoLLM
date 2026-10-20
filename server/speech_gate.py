"""Speech-anchored attention gating for Qwen2.5-Omni Thinker (single forward pass, no training, no re-encoding).

Qwen2.5-Omni gives every audio and video token a temporal rope id (25 ids per second, shared clock).
Given a time window [s0, s1] (seconds), we find the video tokens whose temporal patch overlaps the window and,
for every text query position AFTER the media block (question + assistant prefix), we add a bias to the attention
logits of video keys OUTSIDE the window:  hard = -inf (attention-level crop), soft = -beta.
Media tokens themselves attend as usual, so the visual and audio representations are unchanged; only what the
question/answer positions read from is steered.
"""
import torch


def media_layout(model, proc, inputs):
    """Return position_ids (3,1,L), per-token time in seconds for video tokens, and index sets."""
    ids = inputs['input_ids']
    am = inputs['attention_mask']
    afl = inputs['feature_attention_mask'].sum(-1) if 'feature_attention_mask' in inputs else None
    spg = inputs.get('video_second_per_grid')
    pos, _ = model.get_rope_index(ids, None, inputs.get('video_grid_thw'), am, True, afl, spg)
    tok = proc.tokenizer
    vid_id = tok.convert_tokens_to_ids(proc.video_token)
    aud_id = tok.convert_tokens_to_ids(proc.audio_token)
    is_v = (ids[0] == vid_id)
    is_a = (ids[0] == aud_id)
    t = pos[0, 0].float()
    base = t[is_a].min() if is_a.any() else t[is_v].min()
    sec = (t - base) / 25.0                       # temporal rope id -> seconds on the shared clock
    last_media = int(torch.nonzero(is_v | is_a).max())
    return pos, sec, is_v, is_a, last_media


def gated_mask(L, is_v, sec, last_media, window, mode, beta, dtype, device, spg=None, scope='all', is_a=None, block_audio=False):
    """scope='text': only positions after the media block are gated (leaks through media tokens that already read the
    out-of-window frames). scope='all': no position except the out-of-window video tokens themselves may read them,
    i.e. an attention-level crop that keeps positions, audio and the full token sequence."""
    neg = torch.finfo(dtype).min
    m = torch.zeros(L, L, dtype=torch.float32, device=device)
    m = m.masked_fill(torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), 1), float('-inf'))
    if mode != 'none':
        s0, s1 = window
        width = float(spg) if spg is not None else 0.0   # a temporal patch covers [t, t + second_per_grid)
        inside = (sec + width > s0) & (sec < s1)
        out_v = is_v.to(device) & ~inside.to(device)
        rows = (torch.arange(L, device=device) > last_media) if scope == 'text' else ~out_v
        bias = float('-inf') if mode == 'hard' else -float(beta)
        m[rows.unsqueeze(1) & out_v.unsqueeze(0)] += bias
    if block_audio:
        # audio-blind decision: no non-audio position may read audio tokens (audio tokens still read each other)
        aud = is_a.to(device)
        m[(~aud).unsqueeze(1) & aud.unsqueeze(0)] = float('-inf')
    m = m.clamp(min=neg).to(dtype)
    return m.view(1, 1, L, L)


def binding_mask(L, is_v, is_a, sec, dtype, device, chunk_s=2.0, delta=0, restrict=True, bidir=False, base=None):
    """Synchronous audio-visual binding inside the media block (text rows are never changed).
    Qwen2.5-Omni interleaves every 2-s chunk as [video_c, audio_c] under a causal mask, so
      - video_c can never read audio_c (the sound of the SAME moment comes later in the sequence), and
      - audio_c (and video_c) can read the other modality of ALL earlier moments.
    restrict=True : a cross-modal edge (audio<->video) is kept only if the two chunks differ by <= delta
    bidir=True    : additionally let video_c read audio_c' for |c - c'| <= delta even though it comes later (prefill only)
    Within-modality attention and everything involving text tokens keep the original causal pattern."""
    m = torch.zeros(L, L, dtype=torch.float32, device=device) if base is None else base.clone().float()
    if base is None:
        m = m.masked_fill(torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), 1), float('-inf'))
    v, a = is_v.to(device), is_a.to(device)
    ch = torch.floor(sec.to(device) / chunk_s)
    near = (ch.unsqueeze(1) - ch.unsqueeze(0)).abs() <= delta
    cross = (v.unsqueeze(1) & a.unsqueeze(0)) | (a.unsqueeze(1) & v.unsqueeze(0))
    if restrict:
        m[cross & ~near] = float('-inf')
    if bidir:
        later = torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), 1)
        m[(v.unsqueeze(1) & a.unsqueeze(0)) & near & later] = 0.0
    neg = torch.finfo(dtype).min
    return m.clamp(min=neg).to(dtype).view(1, 1, L, L)


def balance_mask(L, is_v, is_a, last_media, dtype, device, strength=1.0):
    """Modality-balanced readout: question/answer rows get an additive bias on audio keys equal to
    strength * log(n_video / n_audio), so that with equal content scores the two streams receive equal total attention
    (Qwen2.5-Omni gives ~144 video tokens vs ~25 audio tokens per second)."""
    m = torch.zeros(L, L, dtype=torch.float32, device=device)
    m = m.masked_fill(torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), 1), float('-inf'))
    v, a = is_v.to(device), is_a.to(device)
    b = strength * float(torch.log(v.sum().float() / a.sum().float().clamp(min=1)))
    rows = torch.arange(L, device=device) > last_media
    m[rows.unsqueeze(1) & a.unsqueeze(0)] += b
    return m.clamp(min=torch.finfo(dtype).min).to(dtype).view(1, 1, L, L)


def bound_last_logits(model, proc, inputs, delta=0, restrict=True, bidir=False):
    pos, sec, is_v, is_a, last_media = media_layout(model, proc, inputs)
    L = inputs['input_ids'].shape[1]
    emb = model.get_input_embeddings().weight
    mask = binding_mask(L, is_v, is_a, sec, emb.dtype, emb.device, delta=delta, restrict=restrict, bidir=bidir)
    kw = {k: v for k, v in inputs.items() if k != 'attention_mask'}
    with torch.no_grad():
        out = model(**kw, attention_mask=mask, position_ids=pos.to(emb.device), use_audio_in_video=True)
    return out.logits[0, -1].float()


def gated_last_logits(model, proc, inputs, window=None, mode='none', beta=0.0, use_audio_in_video=True, scope='all', block_audio=False):
    """Forward pass with explicit 3D rope ids and a 4D additive mask; returns last-position logits (float32)."""
    pos, sec, is_v, is_a, last_media = media_layout(model, proc, inputs)
    L = inputs['input_ids'].shape[1]
    emb = model.get_input_embeddings().weight
    spg = inputs.get('video_second_per_grid')
    spg = float(spg.reshape(-1)[0]) if spg is not None else None
    mask = gated_mask(L, is_v, sec, last_media, window, mode, beta, emb.dtype, emb.device, spg, scope, is_a, block_audio)
    kw = {k: v for k, v in inputs.items() if k != 'attention_mask'}
    with torch.no_grad():
        out = model(**kw, attention_mask=mask, position_ids=pos.to(emb.device), use_audio_in_video=use_audio_in_video)
    return out.logits[0, -1].float(), dict(n_video=int(is_v.sum()), n_audio=int(is_a.sum()), last_media=last_media,
                                           video_sec_range=[float(sec[is_v].min()), float(sec[is_v].max())])
