# -----------------------------------------------------------------------------
# Video time to pose-log time.
#
# Author:  Petr Šopák <Petr.Sopak@vut.cz>
#          Brno University of Technology, CEITEC and FEEC
# Part of: Stability of Localization and Tracking on Meta Quest Devices:
#          Statistics, Sources of Failures and VR App for Identification
# -----------------------------------------------------------------------------
"""Placing the video-derived streams on the pose clock.

The scene descriptors and the manual annotations carry a video time, Time_s,
with 0 at the first decoded frame of the cut video segment. That frame is not
the first pose sample: the cut videos start at the first key frame after the
cut point, up to several seconds in, and the Quest 2 stream was recorded on a
laptop, whose clock drifts against the headset's. The offset between the two
clocks was therefore measured rather than assumed, by cross-correlating the
frame-to-frame image motion with the angular rate of the head
(scripts/00_video_offsets.py), and is stored in sync/video_offsets.csv:

    dev, month, seg, video_from_s, offset_s, stream_start_s

meaning that from video time video_from_s onwards, video time T corresponds to
pose-log time t0 + offset_s + T, where t0 is the first pose sample of the
segment. One row per segment where the offset is constant; several rows where
it changes within the recording.

The manual annotations were made on the time of the cut file, which starts at
the cut point, whereas the descriptor time starts at the first decoded frame,
stream_start_s later. timeSynchronization.m matched the two by their first
samples, so an annotation row at video time T describes the frame at T minus
stream_start_s; for the annotations the offset is offset_s - stream_start_s.
"""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd

from . import config as C


@lru_cache(maxsize=1)
def _table() -> pd.DataFrame:
    return pd.read_csv(C.VIDEO_OFFSETS)


@lru_cache(maxsize=None)
def log_start(dev: str, month: str, seg: int, root: str = None) -> float:
    """Time of the first pose sample of a segment, on the pose clock."""
    p = (Path(root or C.DATA_ROOT) / "Data_raw" / dev / "Data" / month
         / f"measHMD_{seg}.csv")
    return float(pd.read_csv(p, sep=";", skiprows=2, header=None, usecols=[0],
                             nrows=1).iloc[0, 0])


def video_to_log(dev: str, month: str, seg: int, t_video: np.ndarray,
                 root: str = None, kind: str = "descriptors") -> np.ndarray:
    """Pose-clock time of every video time in t_video.

    kind = "descriptors" for sync_video_*, "annotations" for sync_anotation_*.
    """
    T = _table()
    r = T[(T.dev == dev) & (T.month == month) & (T.seg == seg)].sort_values("video_from_s")
    if r.empty:
        raise KeyError(f"no measured video offset for {dev} {month} seg {seg}")
    t_video = np.asarray(t_video, float)
    k = np.searchsorted(r.video_from_s.to_numpy(), t_video, side="right") - 1
    k = np.clip(k, 0, len(r) - 1)
    off = r.offset_s.to_numpy()[k]
    if kind == "annotations":
        off = off - r.stream_start_s.to_numpy()[k]
    return log_start(dev, month, seg, root) + off + t_video
