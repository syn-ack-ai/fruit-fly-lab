"""
The native multi-core engine (native/) must be the same model as the Python
engine, spike for spike, and the pipelined session must be the same closed
loop as the serial one. Skipped if the native library has not been built
(`make -C native`).
"""
from __future__ import annotations

import numpy as np
import pytest

from brain.neurons.registry import load_connectome
from native import lif_native
from simulation.engine.lif_engine import LIFEngine
from simulation.engine.session import Session
from tools.prng import Mulberry32

pytestmark = pytest.mark.skipif(not lif_native.available(),
                                reason="native engine not built (make -C native)")


@pytest.fixture(scope="module")
def c():
    return load_connectome()


def _drive(c):
    return np.sort(c.by_cell_types(["LC4", "LPLC2"])["idx"].to_numpy())


def _pair(conn, seed, threads):
    py = LIFEngine(conn, seed=seed)
    py.rng = Mulberry32(seed)                 # the PRNG both engines share
    nat = lif_native.NativeLIFEngine.from_connectome(conn, seed=seed, threads=threads)
    return py, nat


def _assert_same_run(py, nat, steps):
    for s in range(steps):
        a, b = py.step(), nat.step()
        assert np.array_equal(a, b), "spikes differ at step %d" % s
    assert np.array_equal(py.spike_counts, nat.spike_counts)
    # bitwise float32 state, not just spikes
    assert np.array_equal(py.v.view(np.uint32), nat.v.view(np.uint32))
    assert np.array_equal(py.g.view(np.uint32), nat.g.view(np.uint32))


def test_matches_python_engine_deterministic(c):
    """Whole brain, no PRNG: 60 real LC4/LPLC2 cells given a large charge."""
    py, nat = _pair(c, seed=3, threads=3)
    for e in (py, nat):
        e.g[_drive(c)[:60]] = 1600.0
    nat.wake_all()
    _assert_same_run(py, nat, 600)
    assert py.spike_counts.sum() > 0


@pytest.mark.parametrize("silence", [False, True])
def test_matches_python_engine_poisson(c, silence):
    """Whole brain, 150 Hz Poisson drive on the real looming detectors."""
    drive = _drive(c)
    py, nat = _pair(c, seed=7, threads=3)
    for e in (py, nat):
        e.set_poisson(drive, 150.0)
        if silence:
            e.silence(drive)
    _assert_same_run(py, nat, 500)
    assert py.spike_counts.sum() > 0


@pytest.mark.parametrize("threads", [1, 2, 3, 4])
def test_thread_counts_on_a_subgraph(c, threads):
    """Odd-sized real subnetwork (tails not a multiple of the SIMD block)."""
    idx = c.by_cell_types(["LPLC2", "LC4", "DNp01", "DNp02", "DNp04"])["idx"].to_numpy()
    keep = set(int(i) for i in idx)
    for i in idx[:40]:
        keep.update(int(j) for j in c.w.getrow(int(i)).indices)
    sg = c.subgraph(np.array(sorted(keep)))
    assert sg.n % 16 != 0
    drive = np.sort(sg.by_cell_types(["LC4", "LPLC2"])["idx"].to_numpy())
    py, nat = _pair(sg, seed=11, threads=threads)
    for e in (py, nat):
        e.set_poisson(drive, 200.0)
    _assert_same_run(py, nat, 400)


def test_reset_reproduces_a_run(c):
    nat = lif_native.NativeLIFEngine.from_connectome(c, seed=5)
    counts = []
    for _ in range(2):
        nat.reset(seed=5)
        nat.set_poisson(_drive(c), 150.0)
        nat.run(60.0)
        counts.append(nat.spike_counts.copy())
    assert np.array_equal(counts[0], counts[1]) and counts[0].sum() > 0


def _session_run(c, pipelined, silence=False):
    s = Session(c, seed=4, engine="native", pipelined=pipelined)
    if silence:
        s.engine.silence(_drive(c))
    s.add_looming(azimuth_deg=30.0, speed_mm_s=500.0, start_distance_mm=60.0)
    frames = []
    for _ in range(40):
        frames.extend(s.advance(5.0))
    for f in frames:
        f.pop("wall_ms")
    return s, frames


def test_pipelined_session_equals_serial(c):
    """Overlapping Python readout with native compute changes nothing."""
    a, fa = _session_run(c, pipelined=True)
    b, fb = _session_run(c, pipelined=False)
    assert np.array_equal(a.engine.spike_counts, b.engine.spike_counts)
    assert fa == fb
    assert a.raster(1000.0) == b.raster(1000.0)
    assert a.engine.spike_counts.sum() > 0


def test_native_session_escape_needs_lc4_lplc2(c):
    """The escape result on the native engine: Giant Fibre fires, and not
    when the real LC4 and LPLC2 populations are silenced."""
    gf = c.by_cell_type("DNp01")["idx"].to_numpy()
    s, _ = _session_run(c, pipelined=True)
    cut, _ = _session_run(c, pipelined=True, silence=True)
    assert s.engine.spike_counts[gf].sum() > 0
    assert cut.engine.spike_counts[gf].sum() == 0
