# -----------------------------------------------------------------------------
# Stage 5: controller visibility when the hands are not held in the camera view.
#
# Author:  Petr Šopák <Petr.Sopak@vut.cz>
#          Brno University of Technology, CEITEC and FEEC
# Part of: Stability of Localization and Tracking on Meta Quest Devices:
#          Statistics, Sources of Failures and VR App for Identification
#          (supplementary repeatability measurement)
#
# Reads   data/measHMD_<DEV>.csv
# Writes  out/tab_controllers.csv   one row per device and hand
#         out/controller_gaps.csv   one row per interval without controller tracking
# -----------------------------------------------------------------------------
"""Controller tracking with the hands carried at the side.

The main study kept both controllers inside the field of view of the headset
cameras throughout, and says so: the controller reliability it reports therefore
describes favourable visibility rather than arbitrary hand placement. In these
recordings the hands were carried at the side, which is the condition the main
study names and does not measure.

The two tracking architectures make a prediction that this stage tests rather
than assumes. The Quest 2 and the Quest 3 localize their controllers from
infrared emitters observed by the headset cameras, so a hand outside that view
cannot be tracked. The Quest Pro controllers carry their own cameras and
estimate their pose without being seen by the headset, so hand placement should
not affect them.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from repeatability import config as C
from repeatability.io import device_logs, load_device

HANDS = ("left", "right")
MIN_GAP_S = 1.0        # shortest interval without tracking that is counted
JUMP_MIN_M = 0.30      # controller displacement between consecutive samples
HEAD_MAX_M = 0.10      # headset displacement over the same interval


def tracked(a, part):
    s = C.PARTS[part]
    return (a[:, s["track"]] == 1) & np.isfinite(a[:, s["pos"]]).all(axis=1)


def gaps(t, ok):
    d = np.diff(np.r_[0, (~ok).astype(int), 0])
    out = []
    for i0, i1 in zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1):
        i1 = min(i1, len(t) - 1)
        if t[i1] - t[i0] >= MIN_GAP_S:
            out.append((float(t[i0]), float(t[i1] - t[i0])))
    return out


def main():
    C.OUT.mkdir(parents=True, exist_ok=True)
    rows, gap_rows = [], []

    for dev, path in device_logs():
        a = load_device(dev, path)[0]
        t = a[:, C.COL_TIME]
        H = a[:, C.PARTS["hmd"]["pos"]]
        okh = tracked(a, "hmd")
        dH = np.linalg.norm(np.diff(H, axis=0), axis=1)
        exposure = float(t[-1] - t[0])

        for hand in HANDS:
            ok = tracked(a, hand)
            P = a[:, C.PARTS[hand]["pos"]]
            off = np.linalg.norm(P - H, axis=1)
            dP = np.linalg.norm(np.diff(P, axis=0), axis=1)
            both = ok[:-1] & ok[1:] & okh[:-1] & okh[1:]
            jump = (dP > JUMP_MIN_M) & (dH < HEAD_MAX_M) & both

            g = gaps(t, ok)
            for t0, d0 in g:
                gap_rows.append(dict(device=dev, hand=hand, t_s=t0, duration_s=d0))

            rows.append(dict(
                device=dev, hand=hand, exposure_s=exposure,
                tracked_frac=float(ok.mean()),
                untracked_pct=float(100.0 * (~ok).mean()),
                gaps=len(g), gap_total_s=float(sum(d for _, d in g)),
                gap_median_s=float(np.median([d for _, d in g])) if g else np.nan,
                gap_max_s=float(max((d for _, d in g), default=np.nan)),
                offset_median_m=float(np.median(off[ok])) if ok.any() else np.nan,
                vertical_offset_median_m=float(np.median((P[:, 1] - H[:, 1])[ok])) if ok.any() else np.nan,
                jumps=int(jump.sum()),
            ))

    T = pd.DataFrame(rows)
    G = pd.DataFrame(gap_rows)
    T.to_csv(C.OUT / "tab_controllers.csv", index=False)
    G.to_csv(C.OUT / "controller_gaps.csv", index=False)

    print("--- controller tracking with the hands carried at the side")
    print(T.round(3).to_string(index=False))
    print()
    for _, r in T.iterrows():
        if r.tracked_frac == 0.0:
            print(f"{C.DEVICE_LABEL.get(r.device, r.device)}, {r.hand}: never reported as "
                  f"tracked at any point in the recording. Nothing about controller "
                  f"reliability can be read from this stream.")
            continue
        print(f"{C.DEVICE_LABEL.get(r.device, r.device)}, {r.hand}: reported untracked for "
              f"{r.untracked_pct:.1f} % of the recording, in {int(r.gaps)} intervals longer "
              f"than {MIN_GAP_S:.0f} s totalling {r.gap_total_s:.0f} s; the controller sat "
              f"{abs(r.vertical_offset_median_m):.2f} m below the headset.")


if __name__ == "__main__":
    main()
