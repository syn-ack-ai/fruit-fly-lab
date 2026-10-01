"""cortex.v0.NumpyCritic (the rover's) equals cortex.v0.Critic (PyTorch) up to
float rounding: the same start, values, TD updates and Adam."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")


def test_numpy_critic_tracks_the_torch_critic(monkeypatch):
    monkeypatch.setenv("FLY_TORCH_DEVICE", "cpu")
    from cortex.v0 import Critic, NumpyCritic
    n_in, rng = 37, np.random.default_rng(0)
    a, b = Critic(n_in, seed=5), NumpyCritic(n_in, seed=5)
    for k, v in a.state_dict().items():
        assert np.array_equal(v.numpy(), b.state_dict()[k])            # the same start
    worst = 0.0
    for step in range(200):
        phi, phi2 = rng.normal(size=n_in).astype(np.float32), rng.normal(size=n_in).astype(np.float32)
        r = float(rng.normal())
        va, vb = a.value(phi), b.value(phi)
        assert va == pytest.approx(vb, rel=1e-4, abs=1e-5)
        la, lb = a.learn(phi, r, phi2), b.learn(phi, r, phi2)
        assert la == pytest.approx(lb, rel=1e-3, abs=1e-6)
        worst = max(worst, max(float(np.abs(v.numpy() - b.state_dict()[k]).max())
                               for k, v in a.state_dict().items()))
    assert worst < 1e-4                                                   # weights stay together


def test_critic_state_round_trip(monkeypatch, tmp_path):
    monkeypatch.setenv("FLY_TORCH_DEVICE", "cpu")
    from cortex.v0 import Critic, NumpyCritic
    a = Critic(9, seed=1)
    b = NumpyCritic(9, seed=2)
    b.load_state_dict(a.state_dict())                                    # torch -> numpy
    x = np.ones(9, np.float32)
    assert b.value(x) == pytest.approx(a.value(x), rel=1e-6)
    c = Critic(9, seed=3)
    c.load_state_dict(b.state_dict())                                    # numpy -> torch
    assert c.value(x) == pytest.approx(a.value(x), rel=1e-6)
