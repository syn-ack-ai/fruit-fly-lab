"""
Annotation extracts from the Janelia MaleCNS v1.0 release that the model reads
alongside the connectome build (brain/connectivity/build_malecns.py), written
to data/derived/malecns/:

  labels_malecns_v1.0.csv.gz          root_id, label: instance; type; FlyWire
                                      type; synonyms -- with the central-complex
                                      glomeruli also written as FlyWire labels
                                      ("EPG(PB08)_L5" -> "EPG_L5", PEN_a -> PEN1,
                                      PEN_b -> PEN2) for brain/navigation/compass.py,
                                      and the feeding-circuit names ("Shiu 2022:
                                      Roundup" ...) for the fly exam
  column_assignment_malecns_v1.0.csv.gz   root_id, hemisphere, type, hex1, hex2,
                                      column_id: the optic-lobe columns
                                      (assignedOlHex1/2) for brain/sensory/retinotopy.py
  dimorphism_malecns_v1.0.csv.gz      root_id, dimorphism, fru_dsx: sex-specific /
                                      dimorphic annotations, protected from the gap
                                      filling in brain/connectivity/merge.py

Only neurons in the built index (traced) are kept; sides come from the index.

    python -m brain.connectivity.malecns_annotations /path/to/malecns_v1.0
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ANNOT = "body-annotations-male-cns-v1.0-minconf-0.5.feather"
CX = {"EPG": "EPG", "PEG": "PEG", "PEN_a": "PEN1", "PEN_b": "PEN2"}


def _cx_label(inst) -> str | None:
    m = re.match(r"^(EPG|PEG|PEN_a|PEN_b)\(PB\w+\)_([LR])(\d)$", str(inst))
    return f"{CX[m.group(1)]}_{m.group(2)}{m.group(3)}" if m else None


def labels(a: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for b, ty, inst, fw, syn in zip(a.bodyId, a.type, a.instance, a.flywireType, a.synonyms):
        parts = [str(x) for x in (_cx_label(inst), inst, ty, fw, syn)
                 if x is not None and str(x) not in ("nan", "None", "")]
        if parts:
            rows.append((int(b), "; ".join(dict.fromkeys(parts))))
    return pd.DataFrame(rows, columns=["root_id", "label"])


def columns(a: pd.DataFrame, index: pd.DataFrame) -> pd.DataFrame:
    t = a.dropna(subset=["assignedOlHex1", "assignedOlHex2"])
    t = t.merge(index[["root_id", "side"]], left_on="bodyId", right_on="root_id", how="inner")
    t = t[t.side.isin(["left", "right"])]
    out = pd.DataFrame({"root_id": t.bodyId.astype(np.int64), "hemisphere": t.side, "type": t.type,
                        "hex1": t.assignedOlHex1.astype(int), "hex2": t.assignedOlHex2.astype(int)})
    out["column_id"] = (out.hex1 * 1000 + out.hex2).astype(np.int32)
    return out


def dimorphism(a: pd.DataFrame) -> pd.DataFrame:
    out = a.rename(columns={"bodyId": "root_id", "fruDsx": "fru_dsx"})[["root_id", "dimorphism", "fru_dsx"]]
    return out[out.dimorphism.notna() | out.fru_dsx.notna()]


def build(src: Path, out_dir: Path) -> dict:
    import pyarrow.feather as pf
    cols = ["bodyId", "type", "instance", "flywireType", "synonyms", "somaSide",
            "assignedOlHex1", "assignedOlHex2", "dimorphism", "fruDsx"]
    a = pf.read_table(src / ANNOT, columns=cols).to_pandas()
    index = pd.read_csv(out_dir / "neuron_index_malecns_v1.0.csv.gz", usecols=["root_id", "side"])
    a = a[a.bodyId.isin(index.root_id)]
    res = {}
    for name, df in (("labels_malecns_v1.0.csv.gz", labels(a)),
                     ("column_assignment_malecns_v1.0.csv.gz", columns(a, index)),
                     ("dimorphism_malecns_v1.0.csv.gz", dimorphism(a))):
        df.to_csv(out_dir / name, index=False)
        res[name] = len(df)
    return res


def main():
    import config
    src = Path(sys.argv[1] if len(sys.argv) > 1 else config.PROJECT_ROOT.parent / "malecns_v1.0")
    print(build(src, config.DERIVED_DIR / "malecns"))


if __name__ == "__main__":
    main()
