"""BASELINE for SVHalluc temporal (TA / TF / CMTB): what does the frozen model answer, and from which modality?
Parameters (set them as globals BEFORE this file, e.g. with jobs/make_job.py, or edit the defaults):
    PER_CELL   items per (task, gold letter); 0 = ALL usable items (983 questions minus clips longer than MAX_DUR)   default 10
    SEED       selection seed                                        default 2026
    SHARD      (k, n): this server takes items k, k+n, ...           default (0, 1)
    CONDS      conditions to run, any of:
               av       audio + video
               vision   video only
               audio    audio only
    OUT_NAME   output file in OUT                                    default baseline.jsonl
Per item and condition two score lists: <cond> = log-probs summed over all rotations of the option order (position-bias free)
and <cond>_canon = log-probs with the options in their original order (the protocol of published plain-accuracy tables).
Analyse with: python analysis/analyze_baseline.py baseline_s1.jsonl [baseline_s2.jsonl ...]"""
PER_CELL = int(globals().get('PER_CELL', 10)); SEED = int(globals().get('SEED', 2026))
SHARD = tuple(globals().get('SHARD', (0, 1))); OUT_NAME = globals().get('OUT_NAME', 'baseline.jsonl')
CONDS = tuple(globals().get('CONDS', ('av', 'vision', 'audio')))


def baseline_item(i, r):
    path, dur, codec = clip_info(r); stem, opts = parse_question(r['question'])
    media = prepare_video(path); rec = dict(dur=round(dur, 2), codec=codec, n_opts=len(opts), block=narrated_block(media['audio'][0]))
    for c in CONDS:
        rec[c], rec[c + '_canon'] = score(media, stem, opts, c, both=True)
    return rec


if PER_CELL > 0: items = balanced_items(PER_CELL, SEED)
else:
    items = [(i, r) for i, r in temporal_rows() if usable(r)]
items = items[SHARD[0]::SHARD[1]]
print('skipped before selection:', len(SKIPPED), SKIPPED[:5], flush=True)
R = run_loop(items, OUT_NAME, baseline_item, show=CONDS)
RESULT = {'n_ok': sum('error' not in r for r in R), 'n': len(R), 'out': OUT_NAME}
