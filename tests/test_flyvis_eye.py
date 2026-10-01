"""robot/flyvis_eye.py: flyvis drives the connectome's optic lobe (option B).
Skipped where flyvis (or its pretrained models) is not installed."""
import numpy as np
import pytest

pytest.importorskip("flyvis")

from robot.flyvis_eye import EYE_CENTRE, OMM_DEG, hexal_directions  # noqa: E402
from robot.retina import PANO_H, PANO_W  # noqa: E402


def test_hexal_directions_mirror_and_centre():
    u = np.array([0, 0, 1, -1])
    v = np.array([0, 1, 0, 0])
    laz, lel = hexal_directions(u, v, "left")
    raz, rel = hexal_directions(u, v, "right")
    assert laz[0] == EYE_CENTRE["left"] and raz[0] == EYE_CENTRE["right"]
    np.testing.assert_allclose(laz, -raz)                  # the eyes are mirror images
    np.testing.assert_allclose(lel, rel)
    assert raz[1] < raz[0] and laz[1] > laz[0]             # +v is anterior on both eyes
    assert rel[2] == -OMM_DEG and rel[3] == OMM_DEG        # +u is down


@pytest.fixture(scope="module")
def encoder():
    import flyvis
    if not (flyvis.results_dir / "flow/0000/000").exists():
        pytest.skip("flyvis pretrained models not downloaded")
    from brain.neurons.registry import load_connectome
    from brain.sensory.retinotopy import load_retinotopy
    c = load_connectome()
    if "T4a" not in set(c.neurons["primary_type"].dropna()):
        pytest.skip("connectome without optic lobe")
    rt = load_retinotopy(c)
    grey = np.full((PANO_H, PANO_W), 0.5, np.float32)
    from robot.flyvis_eye import FlyvisEncoder
    return FlyvisEncoder(c, rt, lambda t_ms: ("grey", grey))


def test_drives_optic_lobe_not_photoreceptors(encoder):
    d = encoder.provenance["drives"]
    assert "T4a" in d and "Mi1" in d and not any(k.startswith("R") and k[1:].isdigit() for k in d)
    assert np.all(np.diff(encoder.indices) > 0)            # unique, ascending


def test_rates_read_only_reused_and_quiet_on_grey(encoder):
    r0 = encoder.rates_hz(0.0)
    assert encoder.rates_hz(5.0) is r0                     # no flyvis step yet: same array
    r1 = encoder.rates_hz(200.0)
    assert not r1.flags.writeable and r1.shape == encoder.indices.shape
    assert r1.min() >= 0 and r1.mean() < 5.0               # grey: at rest
