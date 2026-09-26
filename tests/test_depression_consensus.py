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
