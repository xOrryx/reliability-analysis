"""Differential controller jitter with the piecewise frame alignment.

The first version aligned the controller and headset frames with one yaw per
recording. After a loss of tracking the relation can change (Section 2.4.1), so
the controller is now aligned piece by piece (excursions.aligned_controller).
This script rebuilds the controller windows both ways and writes:

  out/ctrl_rel_piecewise/tab_ctrl_rel_piecewise.csv   table (Table 8 of the paper)
  out/ctrl_rel_piecewise/tab_ctrl_rel_single.csv      same table, one yaw per recording
  out/ctrl_rel_piecewise/coupling.csv                 rank correlation with the headset, both ways
  out/controllers.pkl, out/tab_ctrl_rel.csv            replaced by the piecewise version
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from scipy.stats import spearmanr

from reliability import config as C
from reliability import frames
from reliability.windows import segment_files


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build(piecewise):
    out = []
    for dev, mon, seg, f in segment_files():
        df = frames.controller_relative(f, piecewise=piecewise)
        if not len(df):
            continue
        df["dev"], df["month"], df["seg"] = dev, mon, seg
        df["hop_s"] = df.attrs["hop_s"]
        out.append(df)
    return pd.concat(out, ignore_index=True)


def coupling(W, R, label):
    M = pd.merge(W[["dev", "month", "seg", "t", "hp_pos_hmd", "f_track_hmd", "ui"]],
                 R[["dev", "month", "seg", "t", "rel_left", "rel_right"]],
                 on=["dev", "month", "seg", "t"])
    M = M[(M.ui < 0.5) & (M.f_track_hmd > 0.99)]
    rows = []
    for dev in C.DEVICES:
        g = M[M.dev == dev].dropna(subset=["hp_pos_hmd"])
        v = pd.concat([g.rel_left, g.rel_right])
        h = pd.concat([g.hp_pos_hmd, g.hp_pos_hmd])
        m = v.notna()
        rows.append(dict(alignment=label, dev=dev, rho=float(spearmanr(v[m], h[m])[0])))
    return rows


def main():
    tables = load_script("02_tables")
    O = C.OUT / "ctrl_rel_piecewise"
    O.mkdir(exist_ok=True)
    W = pd.read_pickle(C.OUT / "windows.pkl")
    rows = []
    for label, pw in (("single", False), ("piecewise", True)):
        print("building", label, flush=True)
        R = build(pw)
        T = tables.table_controllers(R)
        T.to_csv(O / f"tab_ctrl_rel_{label}.csv", index=False)
        rows += coupling(W, R, label)
        print(T.to_string(), flush=True)
        if pw:
            R.to_pickle(C.OUT / "controllers.pkl")
            T.to_csv(C.OUT / "tab_ctrl_rel.csv", index=False)
            (C.OUT / "tex").mkdir(exist_ok=True)
            (C.OUT / "tex" / "tab_ctrl_rel.tex").write_text(tables.tex_controllers(T))
    pd.DataFrame(rows).to_csv(O / "coupling.csv", index=False)
    print(pd.DataFrame(rows).to_string())


if __name__ == "__main__":
    main()
