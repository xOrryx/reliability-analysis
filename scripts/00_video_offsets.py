# -----------------------------------------------------------------------------
# Stage 0: measure the offset between each cut video and its pose log.
#
# Author:  Petr Šopák <Petr.Sopak@vut.cz>
#          Brno University of Technology, CEITEC and FEEC
# Part of: Stability of Localization and Tracking on Meta Quest Devices:
#          Statistics, Sources of Failures and VR App for Identification
# -----------------------------------------------------------------------------
"""Stage 0: video time against pose time, measured from the data.

Needs the cut videos (Data_raw/<dev>/Video/<month>/video_<n>_part<m>.mp4) and
ffmpeg/ffprobe on PATH or in FFMPEG_DIR. It is run once; its output,
sync/video_offsets.csv, is committed, so the rest of the pipeline does not need
the videos.

Method. The image moves when the head turns, so the frame-to-frame image change
and the angular rate of the head are two records of the same motion. Both are
put on a 10 Hz grid (video: mean absolute difference of consecutive 64x64
greyscale frames; pose: angular rate from consecutive quaternions) and the lag
maximising their normalised cross-correlation is taken, in 60 s windows with a
30 s hop, over +-8 s. Windows with correlation below MIN_CORR are not used.

The lag is measured against container time 0 (the cut point). The descriptor
time used downstream starts at the first decoded frame, the video stream's
start_time, so offset_s = lag + start_time.

Windows during a loss of tracking are excluded (the reported pose stops moving
while the image does not, so the correlation is meaningless there), and so is
an isolated window whose lag departs from both neighbours by more than
MAX_SPREAD_S. Per segment one offset (the median) is kept when the remaining
windows agree within MAX_SPREAD_S, which holds for the headset-recorded Quest 3
and Quest Pro videos; otherwise the offset is kept per window, smoothed by a
running median over three windows (laptop-recorded Quest 2 videos, whose clock
drifts and jumps against the headset's).

    python scripts/00_video_offsets.py               measure (needs the videos)
    python scripts/00_video_offsets.py --from-windows  rebuild the table from
                                                     sync/video_offsets_windows.csv

Pairing of video parts and logs follows timeSynchronization.m: the k-th
video_*_part*.csv of a device and month in name order belongs to measHMD_k.csv.
"""
import os, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from reliability import config as C

FFMPEG_DIR = Path(os.environ.get("FFMPEG_DIR", ""))
FF = str(FFMPEG_DIR / "ffmpeg") if FFMPEG_DIR.name else "ffmpeg"
FP = str(FFMPEG_DIR / "ffprobe") if FFMPEG_DIR.name else "ffprobe"
FS, N = 10.0, 64
WIN_S, HOP_S, MAXLAG_S = 60.0, 30.0, 8.0
MIN_CORR, MAX_SPREAD_S = 0.30, 1.0
OUT = Path(__file__).resolve().parents[1] / "sync"


def video_motion(path):
    cmd = [FF, "-loglevel", "error", "-i", str(path), "-vf",
           f"fps={FS},scale={N}:{N},format=gray", "-f", "rawvideo", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    fr = np.frombuffer(raw, np.uint8).reshape(-1, N, N).astype(np.float32)
    m = np.abs(np.diff(fr, axis=0)).mean(axis=(1, 2))
    return np.r_[m[0], m]


def stream_start(path):
    out = subprocess.run([FP, "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=start_time", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return float(out.strip())


def pose_motion(path):
    a = pd.read_csv(path, sep=";", skiprows=2, header=None).iloc[:, :C.N_COLS].astype(float).to_numpy()
    t = a[:, C.COL_TIME] - a[0, C.COL_TIME]
    q = a[:, C.PARTS["hmd"]["rot"]]
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    d = np.abs(np.sum(q[:-1] * q[1:], axis=1))
    w = np.degrees(2 * np.arccos(np.clip(d, -1, 1))) / np.maximum(np.diff(t), 1e-3)
    tc = 0.5 * (t[:-1] + t[1:])
    return np.interp(np.arange(0, tc[-1], 1 / FS), tc, w)


def z(x):
    x = np.log1p(x)
    return (x - x.mean()) / (x.std() + 1e-9)


def sliding_lags(v, p):
    W, H, L = int(WIN_S * FS), int(HOP_S * FS), int(MAXLAG_S * FS)
    rows = []
    for s0 in range(0, max(1, min(len(v), len(p)) - W + 1), H):
        vv = z(v[s0:s0 + W])
        best = (-2.0, 0)
        for k in range(-L, L + 1):
            a0 = s0 + k
            if a0 < 0 or a0 + W > len(p):
                continue
            c = float(np.corrcoef(vv, z(p[a0:a0 + W]))[0, 1])
            if c > best[0]:
                best = (c, k)
        rows.append((s0 / FS, best[1] / FS, best[0]))
    return pd.DataFrame(rows, columns=["container_t0_s", "lag_s", "corr"])


def tracked_fraction(log, t0, t1):
    """Fraction of samples in [t0, t1) s after the first one that are tracked."""
    a = pd.read_csv(log, sep=";", skiprows=2, header=None, usecols=[0, C.PARTS["hmd"]["track"]]).to_numpy(float)
    t = a[:, 0] - a[0, 0]
    m = (t >= t0) & (t < t1)
    return float(a[m, 1].mean()) if m.any() else 0.0


def decide(S):
    """Offset rows for one segment from its window lags (see module docstring)."""
    st = float(S.stream_start_s.iloc[0])
    g = S[(S["corr"] >= MIN_CORR) & (S.tracked >= 0.99)].reset_index(drop=True)
    if g.empty:
        g = S.nlargest(1, "corr").reset_index(drop=True)
    if len(g) >= 3:
        l = g.lag_s.to_numpy()
        prev, nxt = np.r_[l[1], l[:-1]], np.r_[l[1:], l[-2]]
        iso = (np.abs(l - prev) > MAX_SPREAD_S) & (np.abs(l - nxt) > MAX_SPREAD_S)
        g = g[~iso].reset_index(drop=True)
    base = dict(dev=S.dev.iloc[0], month=S.month.iloc[0], seg=int(S.seg.iloc[0]),
                stream_start_s=st, windows=len(S), used=len(g))
    if g.lag_s.max() - g.lag_s.min() <= MAX_SPREAD_S:
        return [dict(base, video_from_s=0.0, offset_s=float(g.lag_s.median()) + st)]
    sm = g.lag_s.rolling(3, center=True, min_periods=1).median().to_numpy()
    vfrom = np.maximum(g.container_t0_s.to_numpy() - st, 0.0)
    vfrom[0] = 0.0
    return [dict(base, video_from_s=float(f), offset_s=float(l) + st) for f, l in zip(vfrom, sm)]


def main():
    root = Path(C.DATA_ROOT) / "Data_raw"
    OUT.mkdir(exist_ok=True)
    if "--from-windows" in sys.argv:
        Wn = pd.read_csv(OUT / "video_offsets_windows.csv")
        if "tracked" not in Wn.columns:
            Wn["tracked"] = [tracked_fraction(root / r.dev / "Data" / r.month / f"measHMD_{int(r.seg)}.csv",
                                              r.container_t0_s + r.lag_s,
                                              r.container_t0_s + r.lag_s + WIN_S)
                             for r in Wn.itertuples()]
            Wn.to_csv(OUT / "video_offsets_windows.csv", index=False)
        table = [r for _, S in Wn.groupby(["dev", "month", "seg"], sort=False) for r in decide(S)]
        T = pd.DataFrame(table)[["dev", "month", "seg", "video_from_s", "offset_s",
                                 "stream_start_s", "windows", "used"]]
        T.to_csv(OUT / "video_offsets.csv", index=False)
        print(T.groupby(["dev", "month", "seg"]).offset_s.agg(["size", "min", "max"]).round(2).to_string())
        return
    table, windows = [], []
    for dev in C.DEVICES:
        for mon in C.MONTHS:
            parts = sorted((root / dev / "VideoParameters" / mon).glob("video_*_part*.csv"))
            for k, c in enumerate(parts, start=1):
                mp4 = root / dev / "Video" / mon / (c.stem + ".mp4")
                log = root / dev / "Data" / mon / f"measHMD_{k}.csv"
                if not (mp4.exists() and log.exists()):
                    continue
                st = stream_start(mp4)
                S = sliding_lags(video_motion(mp4), pose_motion(log))
                S[["dev", "month", "seg", "video", "stream_start_s"]] = dev, mon, k, c.stem, st
                S["tracked"] = [tracked_fraction(log, x + l, x + l + WIN_S)
                                for x, l in zip(S.container_t0_s, S.lag_s)]
                windows.append(S)
                table += decide(S)
                print(dev, mon, k, flush=True)
    T = pd.DataFrame(table)[["dev", "month", "seg", "video_from_s", "offset_s",
                             "stream_start_s", "windows", "used"]]
    T.to_csv(OUT / "video_offsets.csv", index=False)
    pd.concat(windows, ignore_index=True).to_csv(OUT / "video_offsets_windows.csv", index=False)
    print(T.groupby(["dev", "month", "seg"]).offset_s.agg(["size", "min", "max"]).round(2).to_string())


if __name__ == "__main__":
    main()
