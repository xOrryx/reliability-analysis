# -----------------------------------------------------------------------------
# Stage 14: the winter indicators on the repeatability recordings.
#
# Author:  Petr Šopák <Petr.Sopak@vut.cz>
#          Brno University of Technology, CEITEC and FEEC
# Part of: Stability of Localization and Tracking on Meta Quest Devices:
#          Statistics, Sources of Failures and VR App for Identification
# -----------------------------------------------------------------------------
"""Stage 14: J_p and J_r of the repeatability recordings (Reviewer 2, R2.1).

The repeatability recordings were made later, on a later Horizon OS build, by
one wearer carrying each headset in turn along one lit route. If the device
ordering of Section 3.1 survives there, it does not depend on the wearers or on
the winter firmware. The recordings are trimmed exactly as in Section 3.7
(repeatability.io.load_device) and then put through the unchanged winter window
code (reliability.windows.segment_windows); the indicator is not
re-implemented. Intervals: block bootstrap over 60 s blocks of the single
recording per device.
Writes out/jitter_repeatability.csv.
"""
import sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from reliability import config as C, windows
from reliability.stats import bootstrap_median_arrays
from repeatability import config as RC
from repeatability.io import device_logs, load_device

RC.DATA_DIR = Path(C.DATA_ROOT) / "repeatability"


def main():
    rows = []
    tmp = Path(tempfile.mkdtemp())
    for dev, path in device_logs():
        a, info, _ = load_device(dev, path)
        f = tmp / f"rep_{dev}.csv"
        with open(f, "w") as fh:
            fh.write("h\nh\n")
        pd.DataFrame(a[:, :C.N_COLS]).to_csv(f, sep=";", header=False, index=False, mode="a",
                                             float_format="%.7f")
        w = windows.segment_windows(f)
        w = w[(w.ui < 0.5) & (w.f_track_hmd > 0.99)]
        w["block"] = ((w.t - w.t.min()) // 60).astype(int)
        for cond, g in (("standing", w[w.speed < C.STILL_MAX]),
                        ("walking", w[(w.speed > C.WALK_MIN) & (w.speed < C.WALK_MAX)])):
            r = dict(dev=dev, condition=cond, windows=len(g), speed=float(g.speed.median()))
            for col, lab, sc in (("hp_pos_hmd", "jp_mm", 10.0), ("hp_rot_hmd", "jr_vec", 1.0)):
                m, lo, hi = bootstrap_median_arrays([b[col].to_numpy() for _, b in g.groupby("block")])
                r[lab], r[lab + "_lo"], r[lab + "_hi"] = m * sc, lo * sc, hi * sc
            rows.append(r)
    T = pd.DataFrame(rows)
    T.to_csv(C.OUT / "jitter_repeatability.csv", index=False)
    pd.set_option("display.width", 220, "display.precision", 2)
    print(T.to_string(index=False))


if __name__ == "__main__":
    main()
