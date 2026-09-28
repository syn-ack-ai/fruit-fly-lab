"""
Tests that the connectome reproduces circuits established by experiment.

These are the strongest evidence that the pipeline is using real biology: they
assert facts about Drosophila that were discovered in wet labs, and check that
they fall out of the FlyWire wiring diagram we loaded. None of these
relationships is hard-coded anywhere in this project. The same facts are
checked on the Janelia MaleCNS (FLY_DATASET=malecns), which names some types
differently and, being a whole CNS, has the leg motor neurons FAFB lacks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
from brain.neurons.labels import functional_group
from brain.neurons.registry import load_connectome
from simulation.engine.lif_engine import LIFEngine


@pytest.fixture(scope="module")
def c():
    return load_connectome()


def _inputs_by_type(c, root_id):
    df = c.inputs(root_id)
    return (df.groupby(df["primary_type"].astype(str))["syn_count"]
              .sum().sort_values(ascending=False))


MALECNS = config.MALE_CNS


class _NativeAsReference:
    """The native engine behind the reference engine's small API
    (set_poisson(idx, rate), run(ms), spike_counts)."""
    def __init__(self, e):
        self.e = e

    def set_poisson(self, idx, rate):
        idx = np.sort(np.asarray(idx, np.int64))
        self.e.set_poisson(idx, np.full(len(idx), float(rate)))

    def run(self, ms):
        self.e.run(ms)

    @property
    def spike_counts(self):
        return self.e.spike_counts


def _engine(c, seed):
    """FAFB: the reference engine, published model (Shiu et al. 2024).
    Male-CNS brains: tested as they run -- the native engine with their
    calibrated gain and dynamics (the reference engine implements only the
    published dynamics, which these brains are not calibrated for)."""
    if not MALECNS:
        return LIFEngine(c, seed=seed)
    from native.lif_native import NativeLIFEngine
    from simulation.engine.session import apply_calibrated_gain, apply_dynamics
    e = NativeLIFEngine.from_connectome(c, seed=seed, threads=4)
    apply_calibrated_gain(e)
    apply_dynamics(e, c, "calibrated")
    return _NativeAsReference(e)


def _idx(c, group):
    if MALECNS:
        # sugar / bitter GRNs mapped from the FlyWire labels (brain/sensory/crossmap.py);
        # the proboscis extension motor neuron is typed MN9 there
        from brain.sensory.modalities import ALL_MODALITIES, resolve_neurons
        if group == "proboscis_motor":
            return c.by_cell_type("MN9")["idx"].to_numpy(dtype=np.int64)
        key = {"sugar_grn": "taste_sugar", "bitter_grn": "taste_bitter"}[group]
        return resolve_neurons(next(m for m in ALL_MODALITIES if m.key == key), c)
    return np.array([c.idx(r) for r in functional_group(group)
                     if int(r) in c._id2idx], dtype=np.int64)


# ===========================================================================
# 1. The escape circuit: LC4 and LPLC2 -> Giant Fibre (DNp01)
#    von Reyn et al. 2017, Nat Neurosci 20:1176; Ache et al. 2019, Curr Biol
#    29:1073. LC4 and LPLC2 are THE two visual inputs driving the Giant Fibre.
# ===========================================================================
def test_lc4_and_lplc2_are_the_top_visual_inputs_to_the_giant_fibre(c):
    for _, gf in c.by_cell_type("DNp01").iterrows():
        byt = _inputs_by_type(c, gf["root_id"])
        top = list(byt.head(5).index)
        assert "LC4" in top, "LC4 missing from top GF inputs (%s side): %s" % (gf["side"], top)
        assert "LPLC2" in top, "LPLC2 missing from top GF inputs (%s side): %s" % (gf["side"], top)
        # LC4 is the single largest cell-type input to the Giant Fibre.
        assert byt.index[0] == "LC4", "largest GF input was %s" % byt.index[0]


def test_giant_fibre_receives_antennal_mechanosensory_input(c):
    """
    The Giant Fibre is multimodal: Johnston's organ neurons (JO-A / JO-B) give
    it an auditory / air-movement channel alongside the visual one.
    Kamikouchi et al. 2009, Nature 458:165.
    """
    jo = {}
    for _, gf in c.by_cell_type("DNp01").iterrows():
        byt = _inputs_by_type(c, gf["root_id"])
        # the MaleCNS splits JO-A / JO-B into subtypes (JO-A2, JO-B1_a, ...)
        jo_a = byt[[t.startswith("JO-A") for t in byt.index]].sum()
        jo_b = byt[[t.startswith("JO-B") for t in byt.index]].sum()
        if MALECNS:
            # MaleCNS: the GF's Johnston's-organ input is JO-B (JO-B1_a/_c), none
            # from JO-A, and lopsided: 656 synapses on the left GF, 23 on the
            # right (checked 2026-09-27; its right antennal input, like its left
            # ORNs, looks less complete). A dataset difference, recorded here.
            jo[gf["side"]] = jo_a + jo_b
            continue
        assert jo_a > 100, "JO-A -> GF too weak on %s (%d)" % (gf["side"], jo_a)
        assert jo_b > 0
    if MALECNS:
        assert max(jo.values()) > 100 and min(jo.values()) > 0, jo
        if config.DATASET_KEY == "merged":             # gap-filled and symmetrised
            assert abs(np.log(jo["left"] / jo["right"])) < np.log(1.05), jo


def test_lc4_and_lplc2_connections_to_giant_fibre_are_excitatory(c):
    for t in ("LC4", "LPLC2"):
        for _, gf in c.by_cell_type("DNp01").iterrows():
            df = c.inputs(gf["root_id"])
            w = df.loc[df["primary_type"].astype(str) == t, "signed_weight"].sum()
            assert w > 0, "%s -> DNp01 came out inhibitory" % t


def test_lplc2_receives_t4_t5_motion_input(c):
    """
    LPLC2 gets its outward-motion selectivity by pooling all four directional
    T4/T5 subtypes. Klapoetke et al. 2017, Nature 551:237.
    """
    idx = c.by_cell_type("LPLC2")["idx"].to_numpy()
    sub = c.w[:, idx].tocoo()
    src = c.neurons["primary_type"].astype(str).to_numpy()[sub.row]
    tot = pd.Series(np.abs(sub.data)).groupby(src).sum()
    for sub_t in ("T4c", "T5b", "T5c", "T5d"):
        assert tot.get(sub_t, 0) > 1000, "%s -> LPLC2 too weak (%s)" % (sub_t, tot.get(sub_t, 0))


# ===========================================================================
# 2. Early vision: R1-6 -> lamina monopolar cells
#    The canonical lamina cartridge. Rister et al. 2007, Neuron 56:155.
# ===========================================================================
def test_photoreceptors_target_the_lamina_monopolar_cells(c):
    idx = c.by_cell_type("R1-R6" if MALECNS else "R1-6")["idx"].to_numpy()
    sub = c.w[idx, :].tocoo()
    tgt = c.neurons["primary_type"].astype(str).to_numpy()[sub.col]
    tot = pd.Series(np.abs(sub.data)).groupby(tgt).sum().sort_values(ascending=False)
    assert set(tot.head(3).index) == {"L1", "L2", "L3"}, list(tot.head(5).index)
    assert tot.head(4).sum() / tot.sum() > 0.9   # overwhelmingly lamina targets


# ===========================================================================
# 3. Feeding: sugar GRNs -> proboscis motor neurons; bitter GRNs do not.
#    This reproduces the headline result of the reference model,
#    Shiu et al. 2024, Nature 634:210.
# ===========================================================================
@pytest.fixture(scope="module")
def feeding(c):
    return {"sugar": _idx(c, "sugar_grn"),
            "bitter": _idx(c, "bitter_grn"),
            "mn": _idx(c, "proboscis_motor")}


def test_functional_label_groups_are_populated(c, feeding):
    assert len(feeding["sugar"]) >= 20
    assert len(feeding["bitter"]) >= 30
    if MALECNS:                                    # MN9 only: the proboscis extension motor neuron
        assert len(feeding["mn"]) == 2
        return
    assert len(feeding["mn"]) >= 50
    sc = c.neurons["super_class"].astype(str).to_numpy()
    assert (sc[feeding["sugar"]] == "sensory").all()
    assert (sc[feeding["mn"]] == "motor").sum() >= 55


def test_sugar_drives_proboscis_motor_neurons(c, feeding):
    e = _engine(c, seed=2)
    e.set_poisson(feeding["sugar"], 150.0)
    e.run(500.0)
    hz = e.spike_counts[feeding["mn"]].sum() / len(feeding["mn"]) / 0.5
    assert hz > 10.0, "proboscis motor neurons only reached %.1f Hz" % hz


def test_bitter_does_not_drive_proboscis_motor_neurons(c, feeding):
    """Bitter suppresses feeding; it must not produce proboscis extension."""
    e = _engine(c, seed=2)
    e.set_poisson(feeding["bitter"], 150.0)
    e.run(500.0)
    assert e.spike_counts[feeding["mn"]].sum() == 0


def test_an_unstimulated_brain_is_silent(c):
    """No input, no spikes. Guards against spurious background activity."""
    e = LIFEngine(c, seed=2)
    e.run(200.0)
    assert int(e.spike_counts.sum()) == 0


# ===========================================================================
# 4. End-to-end: looming activation reaches the Giant Fibre, and cutting the
#    real presynaptic populations abolishes it.
# ===========================================================================
@pytest.fixture(scope="module")
def escape_runs(c):
    gf = c.by_cell_type("DNp01")["idx"].to_numpy()
    lc4 = c.by_cell_type("LC4")["idx"].to_numpy()
    lplc2 = c.by_cell_type("LPLC2")["idx"].to_numpy()
    drive = np.concatenate([lc4, lplc2])

    out = {}
    for name, silenced in (("intact", None), ("cut", drive)):
        e = LIFEngine(c, seed=5)
        e.set_poisson(drive, 150.0)
        if silenced is not None:
            e.silence(silenced)
        e.run(200.0)
        out[name] = e.spike_counts.copy()
    out["gf"], out["drive"] = gf, drive
    return out


def test_looming_populations_drive_the_giant_fibre(escape_runs):
    assert escape_runs["intact"][escape_runs["gf"]].sum() > 0


def test_silencing_lc4_and_lplc2_abolishes_the_giant_fibre_response(escape_runs):
    """The response travels through the connectome, not around it."""
    assert escape_runs["cut"][escape_runs["gf"]].sum() == 0


def test_silenced_neurons_still_spike_but_send_nothing(escape_runs):
    """Silencing removes output only, as in Shiu et al.'s `silence()`."""
    d = escape_runs["drive"]
    assert escape_runs["cut"][d].sum() > 0
    others = np.setdiff1d(np.arange(len(escape_runs["cut"])), d)
    assert escape_runs["cut"][others].sum() == 0


def test_activity_spreads_beyond_the_stimulated_population(escape_runs):
    n_active = int((escape_runs["intact"] > 0).sum())
    assert n_active > len(escape_runs["drive"]), (
        "only %d neurons active; signal did not propagate" % n_active)


# ===========================================================================
# 5. Descending neurons are the brain's output; VNC motor neurons are absent.
# ===========================================================================
def test_no_leg_or_wing_motor_neurons_in_this_brain_dataset(c):
    """
    FlyWire FAFB is brain-only. Everything labelled `motor` innervates head
    structures. If leg/wing motor neurons ever appear, our documented scope
    limit is wrong and must be revisited. (MaleCNS: the opposite -- the leg
    motor neurons are there, leaving through the leg nerves.)
    """
    if MALECNS:
        vm = c.neurons[c.neurons["super_class"].astype(str) == "vnc_motor"]
        nerves = vm["nerve"].astype(str).value_counts()
        for leg_nerve in ("ProLN", "MesoLN", "MetaLN"):   # front, middle, hind leg nerves
            assert nerves.get(leg_nerve, 0) > 50, leg_nerve
        return
    motor = c.neurons[c.neurons["super_class"].astype(str) == "motor"]
    assert len(motor) == 110
    nerves = set(motor["nerve"].astype(str).unique())
    for forbidden in ("ProLN", "MesoLN", "MetaLN", "DProN", "AbN"):
        assert forbidden not in nerves


def test_descending_neuron_population_is_intact(c):
    dn = c.neurons[c.neurons["super_class"].astype(str) == ("descending_neuron" if MALECNS else "descending")]
    assert len(dn) == (1314 if MALECNS else 1305)
    for t in ("DNp01", "DNp02", "DNp04", "DNp09", "DNp11", "DNa01", "DNa02", "MDN"):
        assert len(c.by_cell_type(t)) >= 2, "%s missing" % t
