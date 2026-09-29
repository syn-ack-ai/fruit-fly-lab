"""Short-term depression of output synapses (native engines) and the
bilateral-consensus wiring (simulation/engine/session.py)."""
from __future__ import annotations

import numpy as np
import pytest

from brain.neurons.registry import load_connectome
from native import lif_native
from simulation.engine.session import bilateral_consensus

pytestmark = pytest.mark.skipif(not lif_native.available(), reason="native engine not built")


@pytest.fixture(scope="module")
def c():
    return load_connectome()


def _orns(c):
    t = c.neurons["primary_type"].fillna("").astype(str).to_numpy()
    return np.flatnonzero(np.char.startswith(t.astype(str), "ORN_"))


def _run(c, f, ms=3000):
    """Spike trains per 1 ms (10 steps of 0.1 ms), depletion at the end."""
    e = lif_native.NativeLIFEngine.from_connectome(c, seed=3, threads=2)
    if f is not None:
        e.set_std(f, 893.0)
    e.reset(seed=3)
    e.set_poisson(_orns(c), 20.0)
    steps = int(round(1.0 / e.p.dt))
    spikes = [e.run_collect(steps) for _ in range(ms)]
    d = e.std_depletion()
    e.close()
    return spikes, d


def test_all_zero_depression_is_the_published_model(c):
    a, _ = _run(c, None, 300)
    b, d = _run(c, np.zeros(c.n, np.float32), 300)
    assert d is None
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_depletion_reaches_the_expected_steady_state(c):
    f = np.zeros(c.n, np.float32)
    orn = _orns(c)
    f[orn] = 0.78
    spikes, d = _run(c, f)
    release = 1.0 - d[orn].mean()
    # Poisson firing at rate r: mean release = 1 / (1 + (1 - f) r tau). ORNs fire
    # below the 20 Hz drive (LN -> ORN inhibition), so use their measured rate
    # over the last 2 s.
    late = np.concatenate(spikes[1000:])
    r = np.isin(late, orn).sum() / len(orn) / 2.0
    expect = 1.0 / (1.0 + 0.22 * r * 0.893)
    assert 2.0 < r < 25.0
    assert release == pytest.approx(expect, rel=0.15)
    other = np.setdiff1d(np.arange(c.n), orn)
    assert np.all(d[other] == 0.0)
    # depressed ORN synapses transmit less: fewer downstream spikes than without depression
    base, _ = _run(c, None)
    assert sum(len(x) for x in spikes) < sum(len(x) for x in base)


def test_lazy_decay_matches_the_per_step_reference(c):
    """The engines decay d lazily (when a neuron spikes); the result must match
    decaying every neuron every step, computed here in float64."""
    f = np.zeros(c.n, np.float32)
    orn = _orns(c)
    f[orn] = 0.78
    e = lif_native.NativeLIFEngine.from_connectome(c, seed=3, threads=2)
    e.set_std(f, 893.0)
    e.reset(seed=3)
    e.set_poisson(orn, 20.0)
    ed = np.exp(-e.p.dt / 893.0)
    ref = np.zeros(c.n)
    worst = 0.0
    for k in range(2000):
        s = e.run_collect(1)
        ref *= ed
        s = s[f[s] > 0]
        ref[s] = 1.0 - (1.0 - ref[s]) * 0.78
        if k % 250 == 249:
            worst = max(worst, float(np.abs(e.std_depletion()[orn] - ref[orn]).max()))
    e.close()
    assert worst < 1e-5


def test_reading_depletion_does_not_change_the_run(c):
    f = np.zeros(c.n, np.float32)
    f[_orns(c)] = 0.78
    out, dep = [], []
    for look in (False, True):
        e = lif_native.NativeLIFEngine.from_connectome(c, seed=3, threads=2)
        e.set_std(f, 893.0)
        e.reset(seed=3)
        e.set_poisson(_orns(c), 20.0)
        run = []
        for _ in range(300):
            run.append(e.run_collect(10))
            if look:
                e.std_depletion()
        out.append(run)
        dep.append(e.std_depletion())
        e.close()
    assert all(np.array_equal(x, y) for x, y in zip(*out))
    # the state itself, bit for bit: a reading that decayed d in place would
    # round differently from one decay over the whole gap (review)
    assert np.array_equal(dep[0], dep[1])


def test_calibrated_depression_reaches_the_configured_neurons(c, tmp_path, monkeypatch):
    """apply_dynamics: ORNs at their release fraction; on the merged brain every
    neuron of loop_depression's classes too (nothing else); one recovery time,
    and a conflict between the two is an error (review 2026-09-28)."""
    import json
    import config
    from brain.neurons.registry import canonical_super_class
    from simulation.engine.session import apply_dynamics
    e = lif_native.NativeLIFEngine.from_connectome(c, seed=1, threads=1)
    got = {}
    real = e.set_std
    def spy(f, tau):
        got.update(f=np.asarray(f).copy(), tau=tau)
        real(f, tau)
    e.set_std = spy
    cfg = apply_dynamics(e, c, "calibrated")
    std, loop = cfg["orn_short_term_depression"], cfg.get("loop_depression") or {}
    orn = np.zeros(c.n, bool)
    orn[_orns(c)] = True
    f = got["f"]
    assert np.all(f[orn] == np.float32(std["release_f"]))
    if loop.get("release_f"):
        inl = np.isin(canonical_super_class(c.neurons), loop["classes"])
        assert not (inl & orn).any()
        assert np.all(f[inl] == np.float32(loop["release_f"])) and inl.sum() > 100_000
        assert got["tau"] == loop["tau_ms"] == std["tau_ms"]
    else:
        inl = np.zeros(c.n, bool)
    assert np.all(f[~inl & ~orn] == 0.0)
    # two recovery times cannot both be honoured
    bad = dict(cfg, loop_depression={"release_f": 0.995, "tau_ms": 500.0, "classes": ["descending"]})
    (tmp_path / f"dynamics_clash_{config.DATASET_KEY}.json").write_text(json.dumps(bad))
    monkeypatch.setattr(config, "METADATA_DIR", tmp_path)
    with pytest.raises(ValueError, match="one recovery time"):
        apply_dynamics(e, c, "clash")
    e.close()


def test_bilateral_consensus_equalises_mirror_pairs(c):
    m = bilateral_consensus(c)
    n = c.neurons
    t = n["primary_type"].fillna("").astype(str).to_numpy()
    sd = n["side"].fillna("").astype(str).to_numpy()
    w = c.w.tocsr()
    tot = {}
    for side in ("left", "right"):
        pre = np.flatnonzero((t == "LAL051") & (sd == side))
        post = set(np.flatnonzero((t == "DNa02") & (sd == side)).tolist())
        s = 0.0
        for i in pre:
            for q in range(w.indptr[i], w.indptr[i + 1]):
                if w.indices[q] in post:
                    s += abs(w.data[q]) * m[q]
        tot[side] = s
    assert tot["left"] == pytest.approx(tot["right"], rel=1e-5)
    assert np.all((m >= 0.5) & (m <= 2.0))


def test_calibrated_orn_input_is_bilaterally_balanced(c):
    """Review 2026-09-25: compensation and consensus must not compound. Total
    ORN->PN input per cell, left vs right, per PN type (ipsi/contra release
    split off, since it is deliberately asymmetric per ORN branch)."""
    import json
    import config
    from simulation.engine.session import apply_dynamics
    own = config.METADATA_DIR / ("dynamics_calibrated_%s.json" % config.DATASET_KEY)   # the dataset's own calibration
    cfg = json.loads((own if own.exists() else config.METADATA_DIR / "dynamics_calibrated.json").read_text())
    cfg["orn_pn_lateral_release"]["enabled"] = False
    cfg["orn_short_term_depression"]["enabled"] = False
    path = config.METADATA_DIR / "dynamics_zz_test_balance.json"
    path.write_text(json.dumps(cfg))
    try:
        e = lif_native.NativeLIFEngine.from_connectome(c, seed=1, threads=1)
        apply_dynamics(e, c, "zz_test_balance")
        mult = e.plastic_multipliers().copy()
        e.close()
    finally:
        path.unlink()
    n = c.neurons
    t = n["primary_type"].fillna("").astype(str).to_numpy()
    sd = n["side"].fillna("").astype(str).to_numpy()
    w = c.w.tocsr()
    pre = np.repeat(np.arange(c.n), np.diff(w.indptr))
    post = w.indices
    is_orn = np.char.startswith(t.astype(str), "ORN_")
    is_pn = (n["class"].fillna("").astype(str) == "ALPN").to_numpy()
    eff = np.abs(w.data) * mult
    m = is_orn[pre] & is_pn[post]
    logs = []
    for ty in sorted(set(t[is_pn])):
        nl, nr = ((t == ty) & (sd == "left")).sum(), ((t == ty) & (sd == "right")).sum()
        L = m & (t[post] == ty) & (sd[post] == "left")
        R = m & (t[post] == ty) & (sd[post] == "right")
        if nl and nr and eff[L].sum() > 0 and eff[R].sum() > 0:
            logs.append(abs(np.log((eff[L].sum() / nl) / (eff[R].sum() / nr))))
    assert np.median(logs) < 0.03          # raw FlyWire: 0.18; the compounding bug: 0.13


def test_escape_jump_turns_away_from_the_threat():
    from fly.body.foraging_body import ForagingBody
    b = ForagingBody(neural=True, seed=1)
    b.state.heading_deg = 90.0
    b._takeoff(0.0, laterality=+1.0, directed=True, why="test")   # right escape DNs more active
    assert b.state.heading_deg == 180.0                           # counter-clockwise = left = away
