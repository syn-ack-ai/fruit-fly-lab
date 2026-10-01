"""Session._rates_at: when only some encoders' rates change (robot/eye.py,
every 10 ms) the merged rates are rewritten in place of a full merge: the
same values, the same targets (and the native engine's read-only index fast
path)."""
import numpy as np

from simulation.engine.session import Session, _freeze


class Enc:
    def __init__(self, idx, rates):
        self.indices = np.asarray(idx, np.int64)
        self.r = _freeze(np.asarray(rates, np.float64))

    def rates_hz(self, t_ms, stim=None):
        return self.r

    def state(self, t_ms):
        return {}


def full_merge(encs):
    out = {}
    for e in encs:
        for i, r in zip(e.indices, e.r):
            out[int(i)] = r if int(i) not in out else max(out[int(i)], r)
    k = sorted(out)
    return np.array(k), np.array([out[i] for i in k])


def bare():
    s = Session.__new__(Session)
    s.encoders = []
    return s


def test_incremental_merge_matches_full():
    rng = np.random.default_rng(0)
    a = Enc([5, 1, 9], [1.0, 2.0, 3.0])
    b = Enc([2, 9, 7], [4.0, 0.5, 6.0])           # shares 9 with a
    eye = Enc([30, 10, 20, 40], rng.uniform(0, 9, 4))
    s = bare()
    s.encoders = [(a, None), (b, None), (eye, None)]
    first = s._rates_at(0.0)
    for k in range(5):
        eye.r = _freeze(rng.uniform(0, 9, 4))     # only the eye changes
        idx, r = s._rates_at(float(k + 1))
        ei, er = full_merge([a, b, eye])
        assert idx is first[0]
        np.testing.assert_array_equal(idx, ei)
        np.testing.assert_array_equal(r, er)
        assert not r.flags.writeable
    b.r = _freeze(np.array([4.0, 8.0, 6.0]))      # a shared neuron changes: the full merge
    idx, r = s._rates_at(9.0)
    np.testing.assert_array_equal(r, full_merge([a, b, eye])[1])


def test_unchanged_returns_same_object():
    s = bare()
    a, b = Enc([3, 1], [1.0, 2.0]), Enc([2], [3.0])
    s.encoders = [(a, None), (b, None)]
    r1 = s._rates_at(0.0)
    assert s._rates_at(1.0) is r1
