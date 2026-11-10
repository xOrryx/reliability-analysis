# -----------------------------------------------------------------------------
# Stage 1. Reads the raw logs and writes every intermediate table.
#
# Author:  Petr Šopák <Petr.Sopak@vut.cz>
#          Brno University of Technology, CEITEC and FEEC
# Part of: Stability of Localization and Tracking on Meta Quest Devices:
#          Statistics, Sources of Failures and VR App for Identification
# -----------------------------------------------------------------------------
"""Stage 1: everything that reads raw data.

Produces the intermediate tables the rest of the pipeline consumes. Re-running is
idempotent: delete out/ and every downstream number is rebuilt from the logs.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from reliability import config as C, windows, frames, scene, annotations, excursions

C.OUT.mkdir(exist_ok=True)


def main():
    print("windowing pose logs ...", flush=True)
    W = windows.build()
    W.to_pickle(C.OUT / "windows.pkl")
    print(f"  {len(W)} windows, {W.groupby(['dev','month','seg']).ngroups} recordings, "
          f"{W.hop_s.iloc[0]*len(W):.0f} s nominal")

    print("episodes of tracking loss ...", flush=True)
    E = windows.failure_episodes(W)
    E.to_csv(C.OUT / "episodes.csv", index=False)
    print(f"  {len(E)} episodes, {E.dur_s.sum():.0f} s")

    print("controller coordinate frames ...", flush=True)
    F = frames.build_frames()
    F.to_csv(C.OUT / "frames.csv", index=False)
    print(f"  {F.yaw_deg.notna().sum()} fits, "
          f"yaw values {sorted(set(F.yaw_deg.round(0).dropna()))[:6]}")

    print("differential controller jitter ...", flush=True)
    R = frames.build_relative()
    R.to_pickle(C.OUT / "controllers.pkl")

    print("controller excursions ...", flush=True)
    J, X = frames.build_excursions()
    J.to_csv(C.OUT / "controller_jumps.csv", index=False)
    X.to_csv(C.OUT / "controller_events.csv", index=False)
    print(f"  {len(J)} single-sample jumps, {len(X)} merged events (first definition)")

    print("controller excursions, revised definition ...", flush=True)
    X2, F2 = excursions.build()
    X2.to_csv(C.OUT / "controller_events_v2.csv", index=False)
    F2.to_csv(C.OUT / "frames_v2.csv", index=False)
    print(f"  {len(X2)} events, {int(X2.both_hands.sum())} with both hands, "
          f"{int(X2.recovered.sum())} recovered; {len(F2)} frame fits")

    print("scene descriptors ...", flush=True)
    S = scene.build(W)
    S.to_pickle(C.OUT / "scene.pkl")
    print(f"  {len(S)} windows with a scene description, {int(S.FAIL.sum())} with loss")

    print("manual annotations ...", flush=True)
    A = annotations.build(W)
    if len(A):
        A.to_pickle(C.OUT / "annotations.pkl")
        print(f"  {len(A)} windows, {A.shape[1]-6} attributes")
    else:
        print("  no annotation files found; the subjective sweep will be skipped")


if __name__ == "__main__":
    main()
