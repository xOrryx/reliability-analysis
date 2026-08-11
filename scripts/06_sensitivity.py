# -----------------------------------------------------------------------------
# Stage 6. Device ordering against the high-pass cut-off frequency.
#
# Author:  Petr Šopák <Petr.Sopak@vut.cz>
#          Brno University of Technology, CEITEC and FEEC
# Part of: Stability of Localization and Tracking on Meta Quest Devices:
#          Statistics, Sources of Failures and VR App for Identification
# -----------------------------------------------------------------------------
"""Stage 6: is the device ordering an artefact of the chosen cut-off frequency?

Rebuilds the window table at several cut-offs and reports the walking median per
device, the standing median, and their ratio. The 15 Hz row is the control for
residual gait: it sits close to the Nyquist frequency of the recordings, so if
the walking-to-standing ratio survived there, the ratio is not gait leaking
through the filter.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from reliability import config as C, windows

CUTOFFS = (3.0, 5.0, 8.0, 10.0, 15.0)
MM = 10.0

rows = []
for fc in CUTOFFS:
    W = windows.build(fc=fc)
    w = W[(W.month == "February") & (W.ui < 0.5) & (W.f_track_hmd > 0.99)]
    r = {"fc_hz": fc}
    for dev in C.DEVICES:
        g = w[w.dev == dev]
        st = np.nanmedian(g[g.speed < C.STILL_MAX].hp_pos_hmd) * MM
        wk = np.nanmedian(g[g.speed > C.WALK_MIN].hp_pos_hmd) * MM
        r[f"{dev}_standing"] = round(float(st), 3)
        r[f"{dev}_walking"] = round(float(wk), 3)
        r[f"{dev}_ratio"] = round(float(wk / st), 2)
    rows.append(r)
    print(r, flush=True)

T = pd.DataFrame(rows)
T.to_csv(C.OUT / "fc_sensitivity.csv", index=False)
order = ["".join(sorted(C.DEVICES, key=lambda d: r[f"{d}_walking"])) for r in rows]
print("\nordering by walking jitter at each cut-off:", set(order))
