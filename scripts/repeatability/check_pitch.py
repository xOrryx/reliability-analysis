# -----------------------------------------------------------------------------
# Field utility: what elevation is the headset actually sitting at?
#
# Author:  Petr Šopák <Petr.Sopak@vut.cz>
#          Brno University of Technology, CEITEC and FEEC
# Part of: Stability of Localization and Tracking on Meta Quest Devices:
#          Statistics, Sources of Failures and VR App for Identification
#          (supplementary repeatability measurement)
#
# Usage   python scripts/check_pitch.py <measHMD_*.csv> [target_deg]
#
# Not part of the pipeline. Point it at a short log — half a minute of standing
# still, looking at something at eye level is enough — and it says how the
# headset is sitting, so the strap can be adjusted against a number instead of a
# guess. With a target given it also says which way to move the rear cradle.
# -----------------------------------------------------------------------------
"""Report the elevation of the headset's forward axis from a short recording."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from repeatability import config as C
from repeatability.io import head_pitch_deg, load_log

HEAD_RADIUS_M = 0.10   # rear cradle to the centre of rotation, for the hint below


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__ + "\n\nusage: check_pitch.py <log.csv> [target_deg]")
    path = Path(sys.argv[1])
    target = float(sys.argv[2]) if len(sys.argv) > 2 else None

    a = load_log(path)
    t = a[:, C.COL_TIME]
    p = head_pitch_deg(a)
    P = a[:, C.PARTS["hmd"]["pos"]]
    v = np.r_[0.0, np.linalg.norm(np.diff(P[:, [0, 2]], axis=0), axis=1) / np.diff(t)]

    print(f"{path.name}: {t[-1]-t[0]:.0f} s, {len(t)} samples")
    print(f"  pitch  median {np.median(p):+6.1f} deg   "
          f"IQR {np.percentile(p,25):+.1f} to {np.percentile(p,75):+.1f}   "
          f"p5 {np.percentile(p,5):+.1f}  p95 {np.percentile(p,95):+.1f}")

    still, walk = p[v < 0.3], p[v > 0.8]
    if len(still) > 30:
        print(f"  standing still: median {np.median(still):+6.1f} deg  ({len(still)} samples)")
    if len(walk) > 30:
        print(f"  walking:        median {np.median(walk):+6.1f} deg  ({len(walk)} samples)")

    if t[-1] - t[0] > 120:
        thirds = [p[(t - t[0] >= lo) & (t - t[0] < hi)]
                  for lo, hi in ((0, (t[-1]-t[0])/3), ((t[-1]-t[0])/3, 2*(t[-1]-t[0])/3),
                                 (2*(t[-1]-t[0])/3, t[-1]-t[0]+1))]
        m = [np.median(x) for x in thirds if len(x)]
        print("  by thirds:      " + "  ".join(f"{x:+.1f}" for x in m) +
              "   (a trend here means the headset moved on the head)")

    if target is not None:
        d = target - float(np.median(p))
        move_cm = 100.0 * HEAD_RADIUS_M * np.radians(abs(d))
        way = "up" if d < 0 else "down"
        print()
        print(f"  target {target:+.1f} deg, currently off by {d:+.1f} deg.")
        print(f"  Move the rear cradle about {move_cm:.1f} cm {way} on the back of the "
              f"head: raising the rear lowers the front. Do not compensate with the neck, "
              f"the posture will drift back within minutes.")
        if abs(d) < 1.5:
            print("  That is within a degree and a half of the target, which is inside "
                  "the run-to-run variation of how a headset seats. Good enough.")


if __name__ == "__main__":
    main()
