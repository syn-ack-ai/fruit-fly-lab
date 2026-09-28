"""The complete map (brain/connectivity/merge.py, FLY_DATASET=merged) and the
review findings of 2026-09-27 that must not come back."""
import json

import numpy as np
import pandas as pd
import pytest

import config

MERGED = config.DERIVED_DIR / "merged" / "connectome_merged_v1.npz"
needs_merged = pytest.mark.skipif(not MERGED.exists(), reason="build it: python -m brain.connectivity.merge")


def test_family_splits_at_the_first_separator():
    from brain.connectivity.merge import family
    assert [family(x) for x in ("ORN_VA1v", "JO-B1_a", "JO-ED2_a", "BM_Vib", "BM", "Mi1")] == \
        ["ORN", "JO", "JO", "BM", "BM", "Mi1"]


def test_exam_engines_use_the_calibrated_gain(monkeypatch):
    """Review 2026-09-27: the exam built engines without the dataset gain."""
    from cognition.exam import core
    import simulation.engine.session as session
    seen = []
    monkeypatch.setattr(session, "apply_calibrated_gain", lambda e, relative=1.0: seen.append(relative) or 1.0)
    core.Ctx({"dynamics": "published", "wsyn": 0.85})
    assert seen == [0.85]                        # robustness levels are relative to the calibrated gain


def test_calibrated_gain_follows_the_dataset(monkeypatch):
    from simulation.engine.session import _calibrated_gain
    monkeypatch.delenv("FLY_GAIN", raising=False)
    g = _calibrated_gain()
    if config.DATASET_KEY == "fafb":
        assert g == 1.0
    else:
        f = config.METADATA_DIR / ("calibration_%s.json" % config.DATASET_KEY)
        if not f.exists():
            f = config.METADATA_DIR / "calibration_malecns.json"
        assert g == json.loads(f.read_text())["gain"]


@needs_merged
def test_merged_is_left_right_balanced():
    """Every connection group with >= 50 synapses matches its mirror image per
    postsynaptic cell to within 10% (integer counts leave tiny groups off)."""
    from brain.connectivity.merge import _edges, _load, _ncell, _sides, symmetry_types
    n, w = _load(MERGED, config.DERIVED_DIR / "malecns" / "neuron_index_malecns_v1.0.csv.gz")
    # the identity merge.py mirrors by (sense groups as a whole: their subtypes
    # are typed unevenly per side)
    t = symmetry_types(n.primary_type.fillna("").astype(str).to_numpy())
    sd = _sides(n, w)
    E = _edges(w, t, sd)
    E = E[(E.tp != "") & (E.tq != "") & E.sp.isin(["left", "right"]) & E.sq.isin(["left", "right"])]
    g = E.groupby(["tp", "sp", "tq", "sq"])["c"].sum()
    nc = _ncell(t, sd)
    per = g.to_numpy() / nc.reindex(pd.MultiIndex.from_arrays(
        [g.index.get_level_values(2), g.index.get_level_values(3)])).to_numpy()
    f = {"left": "right", "right": "left"}
    m = pd.Series(per, index=g.index).reindex(pd.MultiIndex.from_arrays(
        [g.index.get_level_values(0), [f[x] for x in g.index.get_level_values(1)],
         g.index.get_level_values(2), [f[x] for x in g.index.get_level_values(3)]])).to_numpy()
    big = ~np.isnan(m) & (g.to_numpy() >= 50)
    r = np.abs(np.log2(per[big] / m[big]))
    assert np.mean(r > np.log2(1.1)) < 0.001, np.mean(r > np.log2(1.1))


@needs_merged
def test_no_protected_type_is_filled():
    """Male-specific / dimorphic cell types are never gap-filled."""
    from brain.connectivity.merge import PROTECT
    man = json.loads((config.METADATA_DIR / "build_manifest_merged.json").read_text())
    n = pd.read_csv(config.DERIVED_DIR / "malecns" / "neuron_index_malecns_v1.0.csv.gz")
    dim = pd.read_csv(config.DERIVED_DIR / "malecns" / "dimorphism_malecns_v1.0.csv.gz")
    prot = set(n.loc[n.root_id.isin(dim.loc[dim.dimorphism.isin(PROTECT), "root_id"]), "primary_type"])
    filled = {g["type"] for g in man["population_gaps"]}
    assert not (filled & prot)


def test_held_out_seed_offset_shifts_exam_seeds(monkeypatch):
    import importlib
    monkeypatch.setenv("FLY_EXAM_SEED_OFFSET", "100")
    import cognition.exam.tests as tests
    importlib.reload(tests)
    try:
        assert tests._seeds(False) == (101, 102, 103, 104)
    finally:
        monkeypatch.delenv("FLY_EXAM_SEED_OFFSET")
        importlib.reload(tests)


@needs_merged
@pytest.mark.skipif(config.DATASET_KEY != "merged", reason="the merged brain's own modality balance")
def test_every_sense_is_balanced_left_right():
    """Review 2026-09-27: one-sided JO subtypes kept 'other JO' 2.8x lopsided
    while the per-group test passed. For every mapped sense (and the filled
    families), the input each RECEIVING cell gets from the left sense organ must
    equal what its mirror cell gets from the right (summed over receiving cell
    types present on both sides), within 5%. Raw totals are not the criterion:
    receiving types present on one side only cannot be mirrored."""
    import pandas as pd
    from brain.connectivity.merge import _load, _ncell, _sides
    from brain.neurons.registry import load_connectome
    from brain.sensory.modalities import ALL_MODALITIES, resolve_neurons
    c = load_connectome()
    n, w = _load(MERGED, config.DERIVED_DIR / "malecns" / "neuron_index_malecns_v1.0.csv.gz")
    sd = _sides(n, w)
    t = n.primary_type.fillna("").astype(str).to_numpy()
    nc = _ncell(t, sd)
    groups = {m.key: resolve_neurons(m, c) for m in ALL_MODALITIES if m.supported}
    for fam in ("ORN_", "JO-", "BM"):
        groups[fam] = np.flatnonzero(np.char.startswith(t.astype(str), fam))
    W = abs(w).tocsr()
    for key, idx in groups.items():
        if len(idx) < 10:
            # e.g. heat: 7 thermosensory neurons with a one-neuron subtype on one
            # side cannot be mirrored (13% off; the robot has no heat sense)
            continue
        per = {}
        for side in ("left", "right"):
            pre = idx[sd[idx] == side]
            sub = W[pre].tocoo()
            df = pd.DataFrame({"tq": t[sub.col], "sq": sd[sub.col], "c": sub.data})
            df = df[df.sq.isin(["left", "right"]) & (df.tq != "")]
            g = df.groupby(["tq", "sq"])["c"].sum()
            per[side] = g / nc.reindex(g.index).to_numpy()
        flip = {"left": "right", "right": "left"}
        L = per["left"]
        R = per["right"].rename(index=flip, level=1)            # right organ -> mirror-image receivers
        both = L.index.intersection(R.index)
        if len(both) == 0 or L[both].sum() == 0:
            continue
        assert abs(np.log(L[both].sum() / R[both].sum())) < np.log(1.05), (key, L[both].sum(), R[both].sum())


def test_body_finds_the_proboscis_motor_neurons():
    """Review 2026-09-27: on male-CNS brains the body's proboscis group was
    empty (FlyWire IDs), so the pet never ate."""
    from brain.motor.descending import DescendingReadout
    from brain.neurons.registry import load_connectome
    c = load_connectome()
    r = DescendingReadout(c)
    t = c.neurons.primary_type.astype(str).to_numpy()
    if config.MALE_CNS:
        assert list(t[r.proboscis_idx]) == ["MN9", "MN9"]        # the proboscis extensor, both sides
    else:
        assert 40 <= len(r.proboscis_idx) <= 80                  # FlyWire's labelled group
