# -----------------------------------------------------------------------------
# Stage 12: controller excursions under the revised definition.
#
# Author:  Petr Šopák <Petr.Sopak@vut.cz>
#          Brno University of Technology, CEITEC and FEEC
# Part of: Stability of Localization and Tracking on Meta Quest Devices:
#          Statistics, Sources of Failures and VR App for Identification
# -----------------------------------------------------------------------------
"""Stage 12: controller excursions, revised (reliability/excursions.py).

Writes out/excursions_v2/events.csv (one row per event), frames.csv (yaw and
residual per piece and hand) and summary.csv (per device and session), and
prints the comparison with the first definition (out/controller_events.csv).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from reliability import config as C, excursions

OUT = C.OUT / "excursions_v2"
OUT.mkdir(parents=True, exist_ok=True)


def summary(E):
    rows = []
    for (dev, mon), g in E.groupby(["dev", "month"]):
        r = g[g.recovered]
        rows.append(dict(dev=dev, month=mon, events=len(g), both_hands=int(g.both_hands.sum()),
                         recovered=int(g.recovered.sum()),
                         median_excursion_m=g.excursion_m.median(), max_excursion_m=g.excursion_m.max(),
                         median_recover_s=r.recover_s.median() if len(r) else np.nan,
                         tracked_at_jump=int(g.tracked_at_jump.sum()),
                         tracked_throughout=int((g.tracked_frac >= 99.999).sum()),
                         tracked_below_50=int((g.tracked_frac < 50).sum()),
                         headset_ok=int((g.headset_tracked_frac >= 99.999).sum())))
    return pd.DataFrame(rows)


def main():
    E, F = excursions.build()
    E.to_csv(OUT / "events.csv", index=False)
    F.to_csv(OUT / "frames.csv", index=False)
    S = summary(E)
    S.to_csv(OUT / "summary.csv", index=False)
    old = pd.read_csv(C.OUT / "controller_events.csv")
    pd.set_option("display.width", 230, "display.precision", 2, "display.max_rows", 80)
    print("frame fits with residual > 1 m:")
    print(F[F.resid_m > 1].to_string(index=False))
    print("\npieces per recording (more than one = re-anchoring):")
    print(F.groupby(["dev", "month", "seg"]).piece.nunique().loc[lambda s: s > 1].to_string())
    print("\nREVISED")
    print(S.to_string(index=False))
    print(f"\nall: {len(E)} events ({int(E.both_hands.sum())} both hands), recovered {int(E.recovered.sum())}, "
          f"median excursion {E.excursion_m.median():.2f} m")
    print("\nOLD: events per device and session")
    print(old.groupby(["dev", "month"]).agg(events=("t", "size"), median_m=("excursion_m", "median"),
                                            at_cap=("recover_s", lambda x: int((x >= 19.9).sum()))).to_string())


if __name__ == "__main__":
    main()
