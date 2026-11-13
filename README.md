# Tracking reliability analysis

Python analysis for *Stability of Localization and Tracking on Meta Quest
Devices: Statistics, Sources of Failures and VR App for Identification*.

Every number in the paper is produced by this code. There is no intermediate file
in the repository that some script did not write: delete `out/` and run
`scripts/run_all.py`, and the tables, the figures and every figure quoted in the
running text are rebuilt from the raw logs.

The video side of the pipeline is the `scene-descriptors` repository:
`video_Objective_extractor.m` computes the per-frame descriptors and
`timeSynchronization.m` puts every stream on the video timeline. This repository
starts from the pose logs and from the files that second script writes.

## Two measurements, kept apart

The paper rests on two measurements that were made months apart, on different
routes and on different firmware, and the code keeps them apart for the same
reason the paper does.

**The winter sessions** are the study proper: three headsets carried
simultaneously along one outdoor route on two winter days, from daylight into
complete darkness. They are what Sections 3.1 to 3.7 report, and they are handled
by the `reliability` package.

**The repeatability measurement** is supplementary, added during revision in
answer to a reviewer. One wearer carried each headset in turn along a straight
170 m stretch and walked it out and back ten times, on a lit route chosen so that
tracking would not be lost. It is what Section 3.8 reports, and it is handled by
the `repeatability` package. It was recorded on a later Horizon OS build than the
winter sessions and under deliberately favourable conditions, so it characterises
the pose the runtime delivers rather than its behaviour in an adverse scene, and
it is never pooled with the winter data.

## What the winter sessions measure

Two quantities, kept apart on purpose:

- the **residual variation of the delivered pose** while tracking is maintained,
  a continuous indicator in millimetres and degrees per second;
- the **loss of tracking**, a binary condition the device itself reports.

They are measured differently, they are in different units, and the analysis
shows they are not two levels of one severity scale, so they are never combined.

## What the repeatability measurement adds

A third quantity the winter sessions cannot yield, because that route was
traversed once and never returned to its start: **how far the reported position
of a fixed physical point moves over the course of a recording.** It comes out at
metres over kilometres, and it grows with the distance walked rather than
scattering about a fixed value, so it is drift and not noise.

## Layout

```
reliability/
  config.py       every constant of the winter analysis; nothing hard-coded elsewhere
  windows.py      pose logs to windowed indicators, and episodes of tracking loss
  frames.py       controller frame correction, differential jitter, excursions
  sync.py         video time to pose-log time, from the measured offsets
  excursions.py   controller excursions with piecewise frame alignment
  scene.py        per-frame descriptors joined onto the pose windows
  annotations.py  manual scene annotations on the same window grid
  detect.py       leave-one-segment-out detection of tracking loss
  stats.py        block bootstrap, partial correlation, sign test, ROC helpers
repeatability/
  config.py       every constant of the repeatability analysis
  io.py           raw pose logs, head pitch, native sampling rate
  trim.py         frame resets and the off-route tail, detected from the trajectory
  passes.py       recording -> ordered list of visits to the two marked points
  drift.py        deviation, increment, leg distance, rigid frame decomposition
  jumps.py        step changes of the reported position and their direction
scripts/
  00_video_offsets.py  offset between each video and its pose log -> sync/
  01_build.py     everything that reads raw data -> out/*.pkl, out/*.csv
  02_tables.py    the manuscript tables -> out/tex/*.tex and out/*.csv
  03_intext.py    every number in the running text -> out/intext.csv
  04_sweep.py     systematic test of scene effects on pose quality -> out/sweep.csv
  05_figures.py   the four figures -> out/figures/
  06_sensitivity.py  the indicator against the high-pass cut-off
  07_supplement.py   Online Resource 1 and 2
  08-21_*.py      analyses of the revised manuscript (detector with whole segments held
                  out, scene before a loss, controller events, vector J_r, speed regimes,
                  frame fits, differential controller jitter); each writes to out/
  run_all.py      all of the above, then the repeatability stages if their data is present
scripts/repeatability/
  01_passes.py    visits to the marks -> out/repeatability/visits.csv
  02_drift.py     deviation, increment, legs, frame transform, and Table 10
  03_jumps.py     step changes, their heading and their cadence
  04_figures.py   drift against distance, route against time, trimming before and after
  05_controllers.py  how much of each recording the controllers were tracked at all
  06_confound.py  head pitch per round trip; whether the devices overlap at all
  check_pitch.py  pitch stability within a recording, run on demand
  run_all.py      the six stages, in order
out_reference/
  the result of one complete run of both, kept so a later run can be diffed
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
    Time_s = 0 is the first decoded video frame; it is placed on the pose clock
    by the offset measured for each recording (sync/video_offsets.csv).

Data_processed/time_sync/header_annotation.csv
Data_processed/time_sync/<device>/<month>/sync_annotation_<segment>.csv
    one row per video frame: a timestamp on the pose clock and the manual
    scene attributes. Optional; without it the subjective sweep is skipped.

repeatability/measHMD_<device>.csv
    the repeatability recordings, one continuous log per device, in the same
    29-column layout. Optional; without them Section 3.8 is not rebuilt and
    scripts/run_all.py says so instead of skipping quietly.
```

Devices are `MQ2`, `MQ3`, `MQPro`; months are `January` and `February`. Scene
descriptors exist for February only.

## Running

```
pip install -r requirements.txt
python scripts/run_all.py
```

About four minutes on a laptop for the winter sessions, most of it in the
cross-validated detector, and about a minute for the repeatability stages. Either
half can also be run on its own:

```
python scripts/repeatability/run_all.py
```

## The decisions that matter

Five choices drive most of the results, and each one changed a conclusion when it
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

**The repeatability figure is quoted from the increment between consecutive
returns to a mark, not from the deviation from the first visit.** The wearer
returns to the same physical point every time, so consecutive deviations are not
independent and a standard deviation over them is not a summary of anything. The
increments are approximately independent, and *n* is the number of round trips,
not the number of one-way legs.

## Two things worth reading before quoting a result

`04_sweep.py` implements the test of whether the scene affects the quality of the
pose. An attribute with fewer than six usable groups cannot reach p < 0.05 by a
sign test at all, and counting such attributes as null results would make the
negative finding look stronger than the design supports. The script reports the
tested, the testable and the passing counts separately, and `out/sweep.csv` holds
the per-attribute effect size so the bound can be read off directly.

`scripts/repeatability/06_confound.py` exists because the repeatability
measurement has a confound that no amount of statistics removes. The elevation of
the headset's forward axis differs between the devices, at a median of −4.7° for
the Quest 3 against −11.5° and −10.2° for the other two, and the ranges the
devices cover over their individual round trips do not overlap at all. Where the
cameras point decides which part of the scene the tracking solution is built on,
so device and camera elevation are completely confounded, and the script says so
rather than fitting a model that would extrapolate into a region where nothing
was measured. The device difference in Section 3.8 is reported as a property of
each headset as it is worn, not of its estimator.

## What the repeatability package deliberately does not do

It does not recompute `J_p` and `J_r` on the repeatability recordings. Those
indicators are defined in `reliability/windows.py`, and implementing the same
high-pass and the same windowing a second time is how two numbers that should
agree start to disagree. If the jitter of those recordings is wanted, put them
through the winter pipeline instead.

## Versions

`v1-submission` is the code of the first submission; `v2-revision` is the code of
the revised manuscript. The revision measures the offset between each video and
its pose log instead of assuming it, aligns the headset and controller frames
between re-anchorings of the headset world, computes the rotational indicator from
the angular velocity vector, evaluates the detector with whole segments held out
and before the onset of a loss, and redefines the controller excursion events.
Library versions are pinned in `requirements.txt`, because the detector results
depend on the scikit-learn version at the level of about 0.02 of area under the
curve.

## Licence and citation

See the manuscript for the citation. The pose logs and the per-frame scene
descriptors are published separately on Zenodo (DOI in the manuscript); the
video recordings are not published.
