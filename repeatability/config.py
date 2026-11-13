# -----------------------------------------------------------------------------
# Every constant of the repeatability analysis, in one place.
#
# Author:  Petr Šopák <Petr.Sopak@vut.cz>
#          Brno University of Technology, CEITEC and FEEC
# Part of: Stability of Localization and Tracking on Meta Quest Devices:
#          Statistics, Sources of Failures and VR App for Identification
#          (supplementary repeatability measurement)
# -----------------------------------------------------------------------------
"""Every constant of the repeatability analysis, in one place.

Nothing in this package hard-codes a numeric threshold anywhere else. If a value
appears in the manuscript, it appears here.
"""
from pathlib import Path

# ----------------------------------------------------------------- paths
# One log per device:  <DATA_DIR>/measHMD_<DEV>.csv
# The repeatability measurement is a separate measurement on separate recordings,
# so it keeps its own data and output directories underneath the shared ones
# rather than mixing its results into those of the winter sessions.
from reliability.config import DATA_ROOT as _DATA_ROOT   # one data root for both packages
DATA_DIR = Path(_DATA_ROOT) / "repeatability"
OUT = Path(__file__).resolve().parents[1] / "out" / "repeatability"

DEVICES = ("MQ2", "MQ3", "MQPro")
DEVICE_LABEL = {"MQ2": "Quest 2", "MQ3": "Quest 3", "MQPro": "Quest Pro"}

# ----------------------------------------------------------------- raw log layout
# column indices in measHMD_*.csv (semicolon separated, two header lines).
# Identical to reliability/config.py; repeated here so this package can be read
# without the other one open.
COL_TIME = 0
COL_UI = 1
PARTS = {
    "hmd":   dict(track=2,  pos=slice(4, 7),   rot=slice(7, 11)),
    "left":  dict(track=11, pos=slice(13, 16), rot=slice(16, 20)),
    "right": dict(track=20, pos=slice(22, 25), rot=slice(25, 29)),
}
N_COLS = 29
AXES = ("X", "Y", "Z")   # Unity convention: Y is the vertical axis

# ----------------------------------------------------------------- route geometry
# The route is walked out and back between two marked points. The along-route
# coordinate is the projection of the horizontal trajectory onto its first
# principal axis, so the analysis does not depend on how the route is oriented
# in the reported frame.
SMOOTH_S = 6.0        # smoothing applied before locating a turnaround, seconds
MIN_TURN_GAP_S = 30.0  # two turnarounds closer than this are one turnaround
MIN_LEG_M = 20.0      # a leg shorter than this is not a traversal of the route
END_WINDOW_S = 2.0    # half-width of the window whose median defines the end position

# A stand is preferred over a turnaround when one is present, because the wearer
# is then stationary at a marked spot rather than pivoting past it.
STILL_MAX = 0.15      # m/s, same threshold as the main analysis
MIN_STAND_S = 5.0     # shortest interval accepted as a stand
STAND_NEAR_TURN_S = 20.0   # a stand this close to a turnaround belongs to it

# Distance between the two marked points, measured with a tape or a laser
# rangefinder, in metres. Set to None when no reference was taken; the scale
# table is then skipped rather than silently computed against nothing.
TAPE_DISTANCE_M = None

# ----------------------------------------------------------------- off-route tail
# A recording may continue after the measurement has ended, because the logging
# cannot always be stopped at the moment the wearer stops walking the route.
# Whatever follows the wearer leaving the route is not a traversal and is cut.
# A reset of the reported frame: the origin is redefined, so positions before it
# and after it are not comparable and cannot lie in one analysis. Recentring the
# view does this, and so does a relocalisation after the tracking has been lost.
# The threshold sits far above the largest correction a device applies to its own
# solution while keeping the frame (about 1.5 m in these recordings) and far
# below the size of an actual reset (tens to hundreds of metres).
RESET_MIN_M = 5.0

CORRIDOR_MAD_K = 6.0    # corridor half-width, in robust deviations of the offset
CORRIDOR_MIN_M = 8.0    # the corridor is never taken narrower than this
MIN_OFFROUTE_S = 5.0    # shortest departure that counts as one
RETURN_MIN_S = 30.0     # a return to the corridor lasting this long is not an end

# Manual override, one entry per device: (t_start, t_end) in the log's own time
# base, either bound may be None. A decision taken by a person belongs here,
# where it is visible, and not inside a script.
TRIM_S = {}

# ----------------------------------------------------------------- pose jumps
# A discontinuity of the delivered pose: the reported position changes by more
# than JUMP_MIN_M between two consecutive samples. JUMP_MIN_M is well above any
# displacement a walking wearer produces within one sampling interval.
JUMP_MIN_M = 0.15
JUMP_REPORT_M = (0.15, 0.20, 0.25, 0.30, 0.40)
HEADING_SAMPLES = 10   # samples averaged to establish the direction of travel
HEADING_COS_MAX = 0.5  # above this a step is treated as ordinary forward motion

# Periodicity test: intervals between jumps are compared with multiples of
# PERIOD_S. Under a uniform null the hit rate is 2*PERIOD_TOL_S / PERIOD_S.
PERIOD_S = 5.0
PERIOD_TOL_S = 0.15
PERIOD_GAP_RANGE_S = (1.0, 120.0)

# ----------------------------------------------------------------- validity
# The tracked share and the panel share of every visit are reported in visits.csv
# (track_frac, ui_frac); no visit is excluded on them, since the route was chosen so
# that tracking was not lost and the reported position is logged with the panel open.
