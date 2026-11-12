"""Walking / standing ratio of the differential controller jitter (Table 8),
compared with the headset ratio of Table 3. Output: out/ctrl_rel_piecewise/ratios.csv"""
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from reliability import config as C

T = pd.read_csv(C.OUT / "tab_ctrl_rel.csv")
J = pd.read_csv(C.OUT / "tab_jitter.csv")
rows = []
for s in ("January", "February"):
    g = T[T.session == s].set_index("condition")
    for d in C.DEVICES:
        st, wk = ast.literal_eval(g.loc["standing", d].replace("nan", "None")), \
                 ast.literal_eval(g.loc["walking", d].replace("nan", "None"))
        if st[0] is None:
            continue
        rows.append(dict(body="controller_rel", session=s, dev=d, standing=st[0],
                         walking=wk[0], ratio=wk[0] / st[0]))
g = J[J.quantity == "HMD position [mm]"].set_index("condition")
for d in C.DEVICES:
    st = ast.literal_eval(g.loc["standing", d]); wk = ast.literal_eval(g.loc["walking", d])
    rows.append(dict(body="hmd", session="February", dev=d, standing=st[0],
                     walking=wk[0], ratio=wk[0] / st[0]))
R = pd.DataFrame(rows)
R.to_csv(C.OUT / "ctrl_rel_piecewise" / "ratios.csv", index=False)
print(R.round(2).to_string())
