#!/usr/bin/env python3
"""Compare C++ vs Python parity outputs.

  compare.py decode  py.json cpp.json
  compare.py pipeline py.json <cpp_out_dir>/sequences/<seq>  (frames_dump.json, <seq>.json, eval_<seq>.json)
Exit code 0 = parity within tolerance.
"""

import json
import sys
from pathlib import Path

import numpy as np

BOX_TOL = 1e-3
REL_TOL = 1e-4


def close_boxes(a, b, tol=BOX_TOL):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return a.shape == b.shape and np.allclose(a, b, atol=tol, rtol=0)


def compare_decode(py, cpp, unordered=False):
    p, c = json.load(open(py)), json.load(open(cpp))
    if unordered:  # tie order is implementation-defined: compare as multisets
        for d in (p, c):
            pairs = sorted(zip(d["scores"], d["boxes"]), key=lambda t: (-t[0], [round(v, 2) for v in t[1]]))
            d["scores"], d["boxes"] = [t[0] for t in pairs], [t[1] for t in pairs]
    ok = len(p["boxes"]) == len(c["boxes"]) and close_boxes(p["boxes"], c["boxes"], 1e-3) and \
        np.allclose(p["scores"], c["scores"], atol=1e-6)
    print(f"[decode] python {len(p['boxes'])} boxes, C++ {len(c['boxes'])} boxes -> {'OK' if ok else 'MISMATCH'}")
    if not ok:
        print("  python:", list(zip(p["scores"], p["boxes"])))
        print("  c++   :", list(zip(c["scores"], c["boxes"])))
    return ok


def compare_pipeline(py, cpp_dir):
    p = json.load(open(py))
    cpp_dir = Path(cpp_dir)
    c_frames = json.load(open(cpp_dir / "frames_dump.json"))
    seq = cpp_dir.name
    c_res = json.load(open(cpp_dir / f"{seq}.json"))["res"]
    c_eval = json.load(open(cpp_dir / f"eval_{seq}.json"))["summary"]
    pf = p["frames"]
    if len(pf) != len(c_frames):
        print(f"[pipeline] frame count differs: python {len(pf)} vs C++ {len(c_frames)}")
        return False

    n_motion_exact = n_motion_frames = 0
    max_strength_rel = 0.0
    first_track_mismatch = None
    for a, b in zip(pf, c_frames):
        if any(a["motion_valid_mask"]) or any(b["motion_valid_mask"]):
            n_motion_frames += 1
            same = a["motion_valid_mask"] == b["motion_valid_mask"] and close_boxes(a["motion_boxes"], b["motion_boxes"])
            n_motion_exact += same
            if same:
                sa, sb = np.asarray(a["motion_strengths"]), np.asarray(b["motion_strengths"])
                denom = np.maximum(np.abs(sa), 1e-6)
                max_strength_rel = max(max_strength_rel, float((np.abs(sa - sb) / denom).max()))
        same_track = (a["track_active"] == b["track_active"] and a["track_ids"] == b["track_ids"]
                      and close_boxes(a["track_boxes_smoothed"], b["track_boxes_smoothed"])
                      and a["tier_for_candidate"] == b["tier_for_candidate"])
        if not same_track and first_track_mismatch is None:
            first_track_mismatch = (a, b)

    print(f"[pipeline] frames: {len(pf)}; frames with motion candidates: {n_motion_frames}; "
          f"identical motion candidate lists: {n_motion_exact}/{n_motion_frames}; "
          f"max strength rel. diff: {max_strength_rel:.2e}")
    ok = first_track_mismatch is None and n_motion_exact == n_motion_frames
    if first_track_mismatch is not None:
        a, b = first_track_mismatch
        print(f"[pipeline] first tracker mismatch at frame {a['frame_index']}:")
        for k in ("track_active", "track_ids", "track_boxes_smoothed", "tier_for_candidate", "motion_boxes"):
            print(f"   {k}: py={a[k]}\n   {' ' * len(k)}  c++={b[k]}")
    else:
        print("[pipeline] tracker state (active/ids/boxes/tiers) identical on every frame")

    res_ok = len(p["res"]) == len(c_res) and all(
        (len(x) == 0 and len(y) == 0) or (len(x) == 4 and len(y) == 4 and close_boxes(x, y)) for x, y in zip(p["res"], c_res))
    print(f"[pipeline] predictions JSON ('res'): {'identical' if res_ok else 'DIFFERENT'}")
    ok = ok and res_ok

    def flat(d, prefix=""):
        for k, v in d.items():
            if isinstance(v, dict):
                yield from flat(v, prefix + k + ".")
            elif isinstance(v, (int, float)):
                yield prefix + k, v

    pm = dict(flat(p["summary"]))
    cm = dict(flat({k: v for k, v in c_eval.items() if k != "curves"}))
    metric_ok = True
    for k, v in pm.items():
        cv = cm.get(k)
        good = cv is not None and abs(cv - v) <= 1e-6 + REL_TOL * abs(v)
        metric_ok &= good
        print(f"   {k:<42} python={v:<12.6g} c++={cv if cv is None else f'{cv:.6g}'}  {'' if good else '<-- DIFF'}")
    print(f"[pipeline] metrics: {'identical' if metric_ok else 'DIFFERENT'}")
    return ok and metric_ok


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode in ("decode", "decode-unordered"):
        good = compare_decode(sys.argv[2], sys.argv[3], unordered=mode == "decode-unordered")
    else:
        good = compare_pipeline(sys.argv[2], sys.argv[3])
    sys.exit(0 if good else 1)
