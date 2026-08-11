# Tracking reliability analysis

Python analysis for *Stability of Localization and Tracking on Meta Quest
Devices: Statistics, Sources of Failures and VR App for Identification*.

Every number in the paper is produced by this code. There is no intermediate file
in the repository that some script did not write: delete `out/` and run
`scripts/run_all.py`, and the tables, the figures and every figure quoted in the
running text are rebuilt from the raw logs.

The video side of the pipeline is a separate repository. This one starts from the
pose logs and from the per-frame descriptor files that the MATLAB stage produces.

## What it computes

Two quantities, kept apart on purpose:

- the **residual variation of the delivered pose** while tracking is maintained,
  a continuous indicator in millimetres and degrees per second;
- the **loss of tracking**, a binary condition the device itself reports.

They are measured differently, they are in different units, and the analysis
shows they are not two levels of one severity scale, so they are never combined.

## Layout

```
reliability/
  config.py       every constant of the analysis; nothing is hard-coded elsewhere
  windows.py      pose logs to windowed indicators, and episodes of tracking loss
  frames.py       controller frame correction, differential jitter, excursions
  scene.py        per-frame descriptors joined onto the pose windows
  annotations.py  manual scene annotations on the same window grid
  detect.py       leave-one-segment-out detection of tracking loss
  stats.py        block bootstrap, partial correlation, sign test, ROC helpers
scripts/
  01_build.py     everything that reads raw data -> out/*.pkl, out/*.csv
  02_tables.py    the manuscript tables -> out/tex/*.tex and out/*.csv
  03_intext.py    every number in the running text -> out/intext.csv
  04_sweep.py     systematic test of scene effects on pose quality -> out/sweep.csv
  05_figures.py   the four figures -> out/figures/
  run_all.py      all of the above, in order
```

## Data layout

Point `reliability/config.py:DATA_ROOT` at a directory holding:

```
Data_raw/<device>/Data/<month>/measHMD_<segment>.csv
    semicolon separated, two header lines, 29 columns:
    time, UI flag, then for the headset and each controller a tracked flag,
    a position and a quaternion. Column indices are in config.PARTS.

Data_processed/time_sync/header_video.csv
Data_processed/time_sync/<device>/February/sync_video_<segment>.csv
    one row per video frame: Time_s and 67 scene descriptors.
    Time_s = 0 is the first sample of the matching pose log.

Data_processed/time_sync/header_annotation.csv
Data_processed/time_sync/<device>/<month>/sync_annotation_<segment>.csv
    one row per video frame: a timestamp on the pose clock and the manual
    scene attributes. Optional; without it the subjective sweep is skipped.
```

Devices are `MQ2`, `MQ3`, `MQPro`; months are `January` and `February`. Scene
descriptors exist for February only.

## Running

```
pip install -r requirements.txt
python scripts/run_all.py
```

About four minutes on a laptop, most of it in the cross-validated detector.

## The decisions that matter

Four choices drive most of the results, and each one changed a conclusion when it
was made differently.

**The jitter is the RMS after a 5 Hz high-pass, not after removing a straight
line.** Within a one-second window the head oscillates vertically at the step
cadence, measured here between 1.6 and 2.0 Hz. The window therefore holds about
two full periods and a fitted line removes almost none of it, so a detrended
indicator measures the wearer's gait rather than the estimator. The detrended
version gave a median of 29 mm while walking, roughly fifty times the value a
robot-driven reference measurement gives for the same device class; the
high-passed version gives 1.8 mm.

**The window is one second of wall-clock time, derived from the measured rate of
each recording.** Fixing it at a sample count assumes a rate the devices do not
have: the measured median is 35.7 Hz, and 45.5 Hz for one session. A fixed count
makes the analysed interval differ between devices, which on this data is enough
to reverse their ranking.

**Confidence intervals resample whole segments, not windows.** Windows overlap by
one half, so window-level resampling treats correlated observations as
independent; on this data it makes the intervals about eight times too narrow.

**Descriptors are aggregated inside a window by the median**, except the
overexposed fraction, which uses the mean. Using the mean throughout costs about
0.03 of area under the curve in the detector.

## Reproducibility of the negative results

`04_sweep.py` implements the test of whether the scene affects the quality of the
pose. Read its docstring before quoting the result: an attribute with fewer than
six usable groups cannot reach p < 0.05 by a sign test at all, and counting such
attributes as null results would make the negative finding look stronger than the
design supports. The script reports the tested, the testable and the passing
counts separately, and `out/sweep.csv` holds the per-attribute effect size so the
bound can be read off directly.

## Licence and citation

See the manuscript for the citation. The recordings themselves are not part of
this repository.
