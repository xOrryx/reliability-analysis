# -----------------------------------------------------------------------------
# Every constant of the analysis, in one place.
#
# Author:  Petr Šopák <Petr.Sopak@vut.cz>
#          Brno University of Technology, CEITEC and FEEC
# Part of: Stability of Localization and Tracking on Meta Quest Devices:
#          Statistics, Sources of Failures and VR App for Identification
# -----------------------------------------------------------------------------
"""Every constant of the analysis, in one place.

Nothing in this package hard-codes a numeric threshold anywhere else. If a value
appears in the manuscript, it appears here.
"""
from pathlib import Path

# ----------------------------------------------------------------- paths
# Root of the data tree: <DATA_ROOT>/Data_raw/<dev>/Data/<month>/measHMD_*.csv
#                       <DATA_ROOT>/Data_processed/time_sync/<dev>/February/sync_video_*.csv
DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
OUT = Path(__file__).resolve().parents[1] / "out"
# Measured offset between video time and pose-log time, per segment (sync.py).
VIDEO_OFFSETS = Path(__file__).resolve().parents[1] / "sync" / "video_offsets.csv"

DEVICES = ("MQ2", "MQ3", "MQPro")
MONTHS = ("January", "February")
DEVICE_LABEL = {"MQ2": "Quest 2", "MQ3": "Quest 3", "MQPro": "Quest Pro"}

# ----------------------------------------------------------------- pose windows
FC_HZ = 5.0        # high-pass cut-off separating gait from estimator noise
ORDER = 4          # Butterworth order, applied forward and backward (zero phase)
WIN_S = 1.0        # analysis window, wall-clock seconds
OVERLAP = 0.5      # window hop as a fraction of the window
MIN_VALID = 0.80   # a window is measurable when this fraction of samples is valid
FAIL_FRAC = 0.20   # a window is a loss of tracking when the tracked fraction is at most this
MIN_BLOCK_S = 5.0  # shortest contiguous valid block the filter is run on
EDGE_S = 0.5       # discarded at each end of a block, filter transient
MIN_EPISODE_S = 1.0

# speed regimes, m/s
STILL_MAX = 0.15
WALK_MIN = 0.80
WALK_MAX = 1.60    # upper bound used only in the scene-effect sweep

# ----------------------------------------------------------------- raw log layout
# column indices in measHMD_*.csv (semicolon separated, two header lines)
COL_TIME = 0
COL_UI = 1
PARTS = {
    "hmd":   dict(track=2,  pos=slice(4, 7),   rot=slice(7, 11)),
    "left":  dict(track=11, pos=slice(13, 16), rot=slice(16, 20)),
    "right": dict(track=20, pos=slice(22, 25), rot=slice(25, 29)),
}
N_COLS = 29

# ----------------------------------------------------------------- controllers
JUMP_MIN_M = 0.30       # controller displacement between consecutive samples
HEAD_MAX_M = 0.10       # headset displacement over the same interval
RECOVER_TOL_M = 0.30    # offset counts as recovered within this of the recording baseline
MIN_EXCURSION_M = 0.30  # an event must take the offset at least this far from the baseline
MAX_RECOVER_S = 20.0    # an event that never recovers within this is truncated

# ----------------------------------------------------------------- scene
MIN_FRAMES_PER_WINDOW = 5
# Aggregation inside a window, following Eq. (7) of the manuscript: the median for
# every descriptor except the overexposed fraction, for which the mean is used.
MEAN_AGGREGATED = ("OverexposedFrac",)

PHOTOMETRIC = ["Brightness", "Contrast", "DynamicRangeP", "Entropy",
               "Focus_LapVar", "Corners", "OverexposedFrac", "UnderexposedFrac"]

# ----------------------------------------------------------------- inference
BOOTSTRAP_N = 5000
SEED = 0
SENS_LEVELS = (0.80, 0.90)
DARK_QUANTILE = 0.25    # "dark windows" control: the darkest quarter by brightness
MIN_GROUPS_SIGN_TEST = 6   # below this, p < 0.05 is unreachable by a sign test
