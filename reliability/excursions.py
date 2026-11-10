# -----------------------------------------------------------------------------
# Controller excursions, revised definition.
#
# Author:  Petr Šopák <Petr.Sopak@vut.cz>
#          Brno University of Technology, CEITEC and FEEC
# Part of: Stability of Localization and Tracking on Meta Quest Devices:
#          Statistics, Sources of Failures and VR App for Identification
# -----------------------------------------------------------------------------
"""Controller excursions, revised.

The first definition (frames.controller_excursions) referenced every event to
one baseline per recording, the median controller-to-headset distance, after a
single yaw correction per recording. Two things break that:

* the relation between the headset frame (world) and the controller frame
  (tracking space) is not constant within a recording: when the headset
  re-anchors its world after a loss of tracking, the controllers stay in the old
  tracking space, so the offset changes by tens to hundreds of metres. A single
  fit then fails (Feb seg 1 Quest 3: -23.5 deg with a 32 m residual instead of
  -45 deg), and every sample after the change looks like an excursion that never
  recovers (the 20 s cap of the old definition);
* the offset of a hand that is not held still drifts by decimetres, so a
  recording-wide baseline is not the place the controller jumped from.

Here
1. the recording is cut where the headset position steps by more than
   RESET_M between consecutive samples, and at the end of every loss of headset
   tracking: after a loss the runtime may re-anchor its world against the
   tracking space even when the headset position itself resumes (Feb seg 1,
   Quest 3: 0 deg and 0 m before a 12 s loss, -45 deg and 148 m after it, with
   a 0.9 m step of the headset), and the yaw and the
   horizontal translation between the two frames are fitted per piece, robustly
   (Procrustes refitted after dropping samples beyond 3 robust deviations);
2. a candidate is, as before, a sample at which a controller moves by more
   than JUMP_MIN_M while the headset moves by less than HEAD_MAX_M; both
   samples must be inside one piece;
3. the departure is measured against the LOCAL baseline, the median offset over
   BASE_S before the candidate, and the event lasts until the offset is back
   within RECOVER_TOL_M of that baseline, for at most MAX_RECOVER_S; events that
   do not return within that are flagged recovered = False instead of being
   given the cap as a recovery time;
4. a candidate of the other hand within PAIR_S whose jump vector agrees within
   PAIR_TOL_M is the same event: both controllers moved together, which points
   at the frame rather than at one controller, and it is reported once with
   both_hands = True;
5. tracked_at_jump (the flag in the jump sample, which is required by the
   candidate rule) and tracked_frac (over the whole event) are both reported.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from . import config as C
from .windows import load_segment, segment_files

RESET_M = 5.0
BASE_S = 5.0
MAX_RECOVER_S = 120.0
PAIR_S = 1.0
PAIR_TOL_M = 0.3
HANDS = {"left": "L", "right": "R"}


def _valid(a, part):
    s = C.PARTS[part]
    return (a[:, s["track"]] == 1) & np.isfinite(a[:, s["pos"]]).all(axis=1)


def _yaw_fit(src, dst):
    a = src - src.mean(axis=0); b = dst - dst.mean(axis=0)
    th = np.arctan2(np.sum(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]),
                    np.sum(a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1]))
    return th


def _rot(xz, th):
    c, s = np.cos(th), np.sin(th)
    return np.column_stack([c * xz[:, 0] - s * xz[:, 1], s * xz[:, 0] + c * xz[:, 1]])


def robust_frame(C_xz, H_xz, iterations=4):
    """Yaw (rad) and translation mapping controller onto head, robust to excursions."""
    keep = np.ones(len(C_xz), bool)
    th, t = 0.0, np.zeros(2)
    for _ in range(iterations):
        th = _yaw_fit(C_xz[keep], H_xz[keep])
        r = _rot(C_xz, th)
        t = np.median(H_xz[keep] - r[keep], axis=0)
        res = np.linalg.norm(r + t - H_xz, axis=1)
        mad = np.median(np.abs(res[keep] - np.median(res[keep]))) * 1.4826
        new = res <= np.median(res[keep]) + 3 * max(mad, 0.05)
        if new.sum() < 0.5 * len(new) or (new == keep).all():
            break
        keep = new
    return th, t, float(np.median(res[keep]))


def pieces(H, okh):
    """Index ranges between possible re-anchorings of the headset world."""
    step = np.r_[0.0, np.linalg.norm(np.diff(H, axis=0), axis=1)]
    resumed = np.r_[False, okh[1:] & ~okh[:-1]]          # first tracked sample after a loss
    cuts = np.where(((step > RESET_M) & okh) | resumed)[0]
    b = np.r_[0, cuts, len(H)]
    return list(zip(b[:-1], b[1:]))


def recording_events(path):
    a = load_segment(path)
    t = a[:, C.COL_TIME]
    H = a[:, C.PARTS["hmd"]["pos"]]
    okh = _valid(a, "hmd")
    piece_id = np.zeros(len(t), int)
    fits, cands = [], []
    for hand in ("left", "right"):
        okc = _valid(a, hand)
        P = a[:, C.PARTS[hand]["pos"]]
        Pc = np.full_like(P, np.nan)
        for k, (i0, i1) in enumerate(pieces(H, okh)):
            piece_id[i0:i1] = k
            m = np.zeros(len(t), bool); m[i0:i1] = True; m &= okh & okc
            if m.sum() < 100:
                continue
            th, tr, res = robust_frame(P[m][:, [0, 2]], H[m][:, [0, 2]])
            xz = _rot(P[i0:i1][:, [0, 2]], th) + tr
            Pc[i0:i1, 0], Pc[i0:i1, 2] = xz[:, 0], xz[:, 1]
            Pc[i0:i1, 1] = P[i0:i1, 1]
            fits.append(dict(hand=HANDS[hand], piece=k, t_from=float(t[i0]), t_to=float(t[i1 - 1]),
                             yaw_deg=float(np.degrees(th)), resid_m=res))
        off = Pc - H
        m = okh & okc & np.isfinite(off).all(axis=1)
        dP = np.r_[0.0, np.linalg.norm(np.diff(Pc, axis=0), axis=1)]
        dH = np.r_[0.0, np.linalg.norm(np.diff(H, axis=0), axis=1)]
        same_piece = np.r_[False, piece_id[1:] == piece_id[:-1]]
        for i in np.flatnonzero(m & same_piece & (dP > C.JUMP_MIN_M) & (dH < C.HEAD_MAX_M)):
            cands.append(dict(hand=hand, i=int(i), t=float(t[i]),
                              vec=Pc[i] - Pc[i - 1], jump_m=float(dP[i])))
        if hand == "left":
            offL, mL = off, m
        else:
            offR, mR = off, m
    OFF = {"left": offL, "right": offR}
    OK = {"left": mL, "right": mR}

    cands.sort(key=lambda c: c["t"])
    events, used, last_end = [], set(), {"left": -1, "right": -1}
    for j, c in enumerate(cands):
        if j in used or c["i"] <= last_end[c["hand"]]:
            continue
        hand, i = c["hand"], c["i"]
        other = "right" if hand == "left" else "left"
        partner = None
        for k in range(j + 1, len(cands)):
            o = cands[k]
            if o["t"] - c["t"] > PAIR_S:
                break
            if o["hand"] == other and np.linalg.norm(o["vec"] - c["vec"]) < PAIR_TOL_M:
                partner = k
                break
        off, ok = OFF[hand], OK[hand]
        pre = (t >= t[i] - BASE_S) & (t < t[i] - 0.05) & ok & (piece_id == piece_id[i])
        if pre.sum() < 10:
            continue
        base = np.median(off[pre], axis=0)
        end = int(min(len(t) - 1, np.searchsorted(t, t[i] + MAX_RECOVER_S)))
        dev_ = np.linalg.norm(off[i:end + 1] - base, axis=1)
        dev_[~ok[i:end + 1] | (piece_id[i:end + 1] != piece_id[i])] = np.nan
        back = np.flatnonzero(dev_[1:] <= C.RECOVER_TOL_M)
        recovered = len(back) > 0
        j_end = i + 1 + back[0] if recovered else end
        seg = slice(i, j_end + 1)
        exc = float(np.nanmax(dev_[: j_end - i + 1])) if np.isfinite(dev_[: j_end - i + 1]).any() else np.nan
        if not (exc >= C.MIN_EXCURSION_M):
            continue
        okc = _valid(a, hand)
        events.append(dict(hand=HANDS[hand], t=float(t[i]), jump_m=c["jump_m"], excursion_m=exc,
                           recovered=bool(recovered),
                           recover_s=float(t[j_end] - t[i]) if recovered else np.nan,
                           both_hands=partner is not None,
                           tracked_at_jump=bool(okc[i]),
                           tracked_frac=100.0 * float(np.mean(okc[seg])),
                           headset_tracked_frac=100.0 * float(np.mean(okh[seg])),
                           piece=int(piece_id[i])))
        last_end[hand] = j_end
        if partner is not None:
            used.add(partner)
            last_end[other] = max(last_end[other], cands[partner]["i"])
    return pd.DataFrame(events), pd.DataFrame(fits)


def build(root=None):
    E, F = [], []
    for dev, mon, seg, f in segment_files(root):
        e, fr = recording_events(f)
        for df, acc in ((e, E), (fr, F)):
            if len(df):
                df["dev"], df["month"], df["seg"] = dev, mon, seg
                acc.append(df)
    return (pd.concat(E, ignore_index=True) if E else pd.DataFrame(),
            pd.concat(F, ignore_index=True) if F else pd.DataFrame())
