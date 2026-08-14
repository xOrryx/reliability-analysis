# -----------------------------------------------------------------------------
# Stage 3: discontinuities of the delivered pose and the cadence they follow.
#
# Author:  Petr Šopák <Petr.Sopak@vut.cz>
#          Brno University of Technology, CEITEC and FEEC
# Part of: Stability of Localization and Tracking on Meta Quest Devices:
#          Statistics, Sources of Failures and VR App for Identification
#          (supplementary repeatability measurement)
#
# Reads   data/measHMD_<DEV>.csv
# Writes  out/jumps.csv        one row per step larger than JUMP_MIN_M
#         out/tab_jumps.csv    counts and cumulative magnitude by threshold
#         out/jump_meta.csv    heading control and periodicity test
# -----------------------------------------------------------------------------
"""Step changes of the reported headset position."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from repeatability import config as C, jumps
from repeatability.io import device_logs, load_device, unpack


def main():
    C.OUT.mkdir(parents=True, exist_ok=True)
    J_all, meta = [], []

    for dev, path in device_logs():
        t, P, ok, ui = unpack(load_device(dev, path)[0])
        J = jumps.find_jumps(dev, t, P, ok, ui)
        J_all.append(J)
        m = dict(device=dev,
                 walking_cos=jumps.walking_heading_cos(t, P, ok, ui),
                 n_jumps=int(len(J)),
                 jump_cos_median=float(J.cos_heading.median()) if len(J) else np.nan,
                 along_travel_frac=float(J.along_travel.mean()) if len(J) else np.nan,
                 vertical_share_median=float(J.vertical_share.median()) if len(J) else np.nan,
                 n_corrections=int((~J.along_travel).sum()) if len(J) else 0,
                 **jumps.periodicity(J[~J.along_travel] if len(J) else J))
        meta.append(m)

    J = pd.concat(J_all, ignore_index=True)
    T = jumps.size_table(J)
    M = pd.DataFrame(meta)
    J.to_csv(C.OUT / "jumps.csv", index=False)
    T.to_csv(C.OUT / "tab_jumps.csv", index=False)
    M.to_csv(C.OUT / "jump_meta.csv", index=False)

    print("--- step changes of the reported position")
    print(T.round(3).to_string(index=False))

    print("\n--- are they lost samples or corrections of the estimate?")
    for _, r in M.iterrows():
        print(f"{r.device}: ordinary walking steps follow the heading with "
              f"cos = {r.walking_cos:.2f}; the {int(r.n_jumps)} jumps do not, "
              f"cos = {r.jump_cos_median:.2f}, and only "
              f"{100*r.along_travel_frac:.0f} % point along travel. "
              f"Median vertical share {r.vertical_share_median:.2f}.")

    print("\n--- cadence of the corrections")
    print("(tested on the off-heading steps only: a step that continues in the "
          "direction of travel\n is motion recorded across one interval, not a "
          "correction, and would only add noise here)")
    for _, r in M.iterrows():
        if not np.isfinite(r.hit_rate):
            print(f"{r.device}: too few intervals to test")
            continue
        print(f"{r.device}: {int(r.n_corrections)} corrections; "
              f"{100*r.hit_rate:.0f} % of the {int(r.n_gaps)} intervals between them "
              f"fall within {C.PERIOD_TOL_S:.2f} s of a multiple of {C.PERIOD_S:.0f} s, "
              f"against {100*r.null_rate:.0f} % expected if they were uniformly placed "
              f"in time ({r.ratio:.1f} times the null).")


if __name__ == "__main__":
    main()
