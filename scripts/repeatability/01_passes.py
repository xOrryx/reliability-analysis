# -----------------------------------------------------------------------------
# Stage 1: segment each recording into visits to the two marked route ends.
#
# Author:  Petr Šopák <Petr.Sopak@vut.cz>
#          Brno University of Technology, CEITEC and FEEC
# Part of: Stability of Localization and Tracking on Meta Quest Devices:
#          Statistics, Sources of Failures and VR App for Identification
#          (supplementary repeatability measurement)
#
# Reads   data/measHMD_<DEV>.csv
# Writes  out/visits.csv      one row per visit to a route end
#         out/recordings.csv  one row per recording
# -----------------------------------------------------------------------------
"""Segment each recording into visits to the two marked route ends."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from repeatability import config as C, passes
from repeatability.io import (device_logs, head_pitch_deg, load_device,
                              native_rate, unpack)


def main():
    C.OUT.mkdir(parents=True, exist_ok=True)
    V_all, R_all, T_all = [], [], []

    for dev, path in device_logs():
        a, tinfo, _ = load_device(dev, path)
        T_all.append(tinfo)
        t, P, ok, ui = unpack(a)
        rate = native_rate(t)
        V = passes.build_visits(dev, t, P, ok, ui, rate)
        V_all.append(V)

        s = passes.along_route(P)
        pitch = head_pitch_deg(a)
        R_all.append(dict(
            device=dev, file=path.name, samples=len(t),
            duration_s=float(t[-1] - t[0]), rate_hz=rate,
            tracked_frac=float(ok.mean()), panel_frac=float(ui.mean()),
            path_m=float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()),
            route_extent_m=float(s.max() - s.min()),
            visits=int(len(V)),
            visits_from_stand=int((V.source == "stand").sum()) if len(V) else 0,
            pitch_median_deg=float(np.median(pitch)),
            pitch_q1_deg=float(np.percentile(pitch, 25)),
            pitch_q3_deg=float(np.percentile(pitch, 75)),
            removed_head_s=tinfo["removed_head_s"],
            removed_tail_s=tinfo["removed_tail_s"],
            resets=tinfo["resets"], trim_source=tinfo["source"],
        ))
        print(f"{dev:6s} {t[-1]-t[0]:7.0f} s  {rate:5.2f} Hz  "
              f"route {s.max()-s.min():6.1f} m  path {R_all[-1]['path_m']:7.0f} m  "
              f"pitch {np.median(pitch):+.1f} deg  "
              f"{len(V)} visits, {R_all[-1]['visits_from_stand']} from a stand")

    if not V_all:
        raise SystemExit(f"no measHMD_<DEV>.csv found in {C.DATA_DIR}")

    V = pd.concat(V_all, ignore_index=True)
    R = pd.DataFrame(R_all)
    V.to_csv(C.OUT / "visits.csv", index=False)
    R.to_csv(C.OUT / "recordings.csv", index=False)
    pd.DataFrame(T_all).to_csv(C.OUT / "trim.csv", index=False)

    for r in T_all:
        if not r["removed_samples"]:
            continue
        parts = []
        if r["removed_head_s"] > 0:
            parts.append(f"{r['removed_head_s']:.0f} s from the start, before the last of "
                         f"{r['resets']} redefinition(s) of the reported frame")
        if r["removed_tail_s"] > 0:
            parts.append(f"{r['removed_tail_s']:.0f} s from the end, where the wearer left "
                         f"the route corridor (half-width "
                         f"{r['corridor_half_width_m']:.1f} m) and reached "
                         f"{r['removed_max_offset_m']:.1f} m off it without traversing again")
        print(f"\n  {r['device']}: removed " + " and ".join(parts) +
              f" ({r['removed_samples']} samples). Source: {r['source']}.")

    print()
    for dev, g in V.groupby("device", sort=False):
        for end, h in g.groupby("end", sort=False):
            print(f"  {dev:6s} mark {end}: {len(h)} visits")
    spread = R["pitch_median_deg"].max() - R["pitch_median_deg"].min()
    if len(R) > 1 and spread > 10.0:
        print(f"\n  warning: the median head pitch differs by {spread:.0f} deg across "
              f"the recordings. The cameras therefore saw different parts of the scene, "
              f"and the devices are not directly comparable without saying so.")

    if (V.source == "turnaround").any():
        n = int((V.source == "turnaround").sum())
        print(f"\n  note: {n} of {len(V)} visits have no stand at the mark; the "
              f"position is the median over +-{C.END_WINDOW_S:.0f} s around the "
              f"turnaround instead, which carries the wearer's overshoot.")


if __name__ == "__main__":
    main()
