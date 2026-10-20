"""Smoke test of a fresh server: GPU, model, data, audio decoding, one score per (task, gold) with both scoring protocols.
    python remote_client.py run jobs/smoke.py
Prints, for one clip of each (task, gold) cell: duration, audio codec, narrated block, the option log-probs (rotation-summed and
original-order-only). Takes a few minutes."""
import collections, numpy as np, torch

print('GPUs:', [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], '| max dur', MAX_DUR)
rows = temporal_rows()
print('temporal rows:', len(rows), dict(collections.Counter((r['task_type_abbr'], r['answer']) for _, r in rows)))
items = balanced_items(1, seed=1)
res = []
for i, r in items:
    path, dur, codec = clip_info(r); stem, opts = parse_question(r['question'])
    media = prepare_video(path)
    blk = narrated_block(media['audio'][0])
    rs, canon = score(media, stem, opts, both=True)
    g = 'ABC'.index(r['answer'])
    res.append(dict(idx=i, task=r['task_type_abbr'], gold=r['answer'], dur=round(dur, 1), codec=codec, block=blk,
                    rot_sum=rs, canonical=canon, rot_ok=int(np.argmax(rs)) == g, canon_ok=int(np.argmax(canon)) == g))
    print(res[-1], flush=True)
print('skipped before selection:', SKIPPED[:5])
RESULT = {'n': len(res), 'codecs': dict(collections.Counter(x['codec'] for x in res))}
