"""Readout used by the foraging body: DNg100 (BDN2) forward walking and
per-type steering signals (Rayshubskiy et al. 2025)."""
import numpy as np
import pytest

from brain.motor.descending import DescendingReadout
from brain.neurons.registry import load_connectome


@pytest.fixture(scope="module")
def connectome():
    return load_connectome()


def test_dng100_forward_and_per_type_steering(connectome):
    ro = DescendingReadout(connectome)
    n = connectome.neurons
    t = n["primary_type"].fillna("").astype(str).to_numpy()
    side = n["side"].astype(str).to_numpy()
    sc = np.zeros(connectome.n)
    sc[t == "DNg100"] = 2                       # 40 Hz over a 50 ms window
    sc[(t == "DNa02") & (side == "left")] = 1   # 20 Hz on the left only
    ch = ro.channels(sc, 50.0)
    assert ch["forward_walk"] > 0.3
    assert abs(ch["hz_DNg100"] - 40.0) < 1e-9
    assert ch["lr_DNa02"] < 0 and "lr_DNa01" in ch


def test_body_steers_transiently_on_dna02():
    from fly.body.foraging_body import ForagingBody
    b = ForagingBody(neural=True, seed=0)
    b._walking = True
    ch = {"lr_DNa02": -0.3, "hz_DNg100": 14.6}   # sustained left-dominant DNa02
    turns = []
    for k in range(3000):                        # 3 s at 1 ms
        b._turn_noise = 0.0
        b.update(1.0, ch, float(k))
        b._walking = True
        turns.append(b.state.turn_rate_deg_s)
    assert max(turns[:400]) > 100                # a brisk left turn at onset ...
    assert abs(turns[-1]) < 20                   # ... that fades (biphasic filter)


def _drive(hz):
    from brain.motor.descending import DescendingReadout
    return DescendingReadout.proboscis_drive_from_hz(hz)


def _extended(train_hz, ms_per_value=1.0, last_ms=None):
    """Fraction of the second half (or the last `last_ms`) of a run the
    proboscis is out, for a scaled proboscis-motor rate given per millisecond."""
    from fly.body.foraging_body import ForagingBody
    b = ForagingBody(neural=True, seed=0, spontaneous_takeoff_per_s=0.0)      # as in Habitat
    out = []
    for k, hz in enumerate(train_hz):
        b.update(ms_per_value, {}, float(k), proboscis_drive=_drive(hz))
        out.append(b.state.proboscis_extension > 0.5)
    return float(np.mean(out[-(last_ms or len(out) // 2):]))


def test_bursty_feeding_holds_the_proboscis_out_and_rest_does_not():
    """2026-09-29: the male brain's two MN9 cells fire in ~250 ms bursts at the
    dock; a 50 ms threshold let the proboscis flicker (ate 16% of dock time)."""
    burst = np.tile(np.r_[np.full(250, 30.0), np.zeros(250)], 12)       # mean 15 Hz, 6 s
    assert _extended(burst) == 1.0
    assert _extended(np.full(6000, 2.0)) == 0.0                         # resting rate
    # the sugar stops: retracted within 2 s
    assert _extended(np.r_[burst, np.zeros(6000)], last_ms=4000) == 0.0


def test_bad_proboscis_drive_does_not_stick():
    """A NaN or saturated drive must not poison the averaged rate (review
    2026-09-29: one NaN stopped the proboscis for good; 1.0 held it for 16 s)."""
    from fly.body.foraging_body import ForagingBody
    b = ForagingBody(neural=True, seed=0, spontaneous_takeoff_per_s=0.0)
    b.update(1.0, {}, 0.0, proboscis_drive=float("nan"))
    b.update(1.0, {}, 1.0, proboscis_drive=1.0)
    assert np.isfinite(b._prob_hz)
    for k in range(4000):
        b.update(1.0, {}, 2.0 + k, proboscis_drive=0.0)
    assert b.state.proboscis_extension < 0.5
    for k in range(3000):
        b.update(1.0, {}, 5000.0 + k, proboscis_drive=_drive(20.0))
    assert b.state.proboscis_extension > 0.5


def test_proboscis_gain_scales_each_brain_to_fafbs_sugar_rate(connectome, monkeypatch):
    import json
    import config
    from brain.motor.descending import PROBOSCIS_REF_SUGAR_HZ, DescendingReadout
    monkeypatch.delenv("FLY_PROBOSCIS_HOLD", raising=False)
    # the reference is FAFB's measured value
    fafb = json.loads((config.METADATA_DIR / "body_readout_fafb.json").read_text())
    assert PROBOSCIS_REF_SUGAR_HZ == fafb["proboscis_sugar_hz"]
    ro = DescendingReadout(connectome)
    if config.DATASET_KEY == "fafb":
        assert ro.proboscis_gain == 1.0
    elif config.DATASET_KEY == "merged":
        assert abs(ro.proboscis_gain - 15.05 / 8.56) < 1e-9                # male MN9 8.56 Hz on sugar
    monkeypatch.setenv("FLY_PROBOSCIS_HOLD", "0")
    assert DescendingReadout(connectome).proboscis_gain == 1.0


def test_fafb_is_the_walking_reference():
    import json
    import config
    import fly.body.foraging_body as fb
    fafb = json.loads((config.METADATA_DIR / "body_readout_fafb.json").read_text())
    assert fb.FAFB_DNG100_REST_HZ == fb.DNG100_REF_REST_HZ == fafb["dng100_rest_hz"]
    if config.DATASET_KEY == "fafb":
        assert fb.DNG100_TAU_SCALE == 1.0


def _walk(ch_of_ms, n):
    from fly.body.foraging_body import ForagingBody
    b = ForagingBody(neural=True, seed=0, spontaneous_takeoff_per_s=0.0)
    beh = []
    for k in range(n):
        b.update(1.0, ch_of_ms(k), float(k))
        beh.append(b.state.behaviour)
    return beh


def test_a_chance_mdn_burst_does_not_walk_backward():
    """2026-09-29: 3 MDN spikes in one 50 ms window (30 Hz on a 2-cell side)
    walked the male robot backward 7.5% of the time; a sustained command
    still does."""
    from brain.motor.descending import CHANNEL_HALF_MAX_HZ as h
    blip = lambda k: {"backward_walk": 30.0 / (30.0 + h) if 1000 <= k < 1050 else 0.0}
    assert not any(x.startswith("walking backward") for x in _walk(blip, 2000))
    held = lambda k: {"backward_walk": 60.0 / (60.0 + h) if k >= 1000 else 0.0}
    assert _walk(held, 2000)[-1].startswith("walking backward")


def test_dng100_above_its_own_rest_starts_walking_bouts():
    """The forward threshold was set on FAFB's rates: a brain's DNg100 is read
    relative to its own resting rate (2026-09-29: the male brain's ~3 Hz
    never crossed FAFB's absolute threshold except in chance bursts)."""
    import fly.body.foraging_body as fb
    fwd = lambda r: _walk(lambda k: {"hz_DNg100": r * fb.DNG100_REST_HZ}, 8000)[-3000:]
    frac = lambda r: np.mean([x.startswith("walking forward") for x in fwd(r)])
    assert frac(2.0) == 1.0
    assert frac(1.0) == 0.0


def test_walking_speed_is_not_lost_to_dng100_shot_noise():
    """2026-09-29: the male brain's two DNg100 cells rest at ~3 Hz and fire in
    bursts; read over FAFB's 0.3 s its speed was mostly noise, and the speed
    limits cut the bursts off (the robot walked ~2/3 as far as FAFB). With the
    smoothing scaled to the spike count, bursts at ~1.3x the resting rate keep
    >= 85% of that pace below the wheels' limit (0.5 m/s = 20 mm/s; on the
    male brain FAFB's 0.3 s kept 74%)."""
    import fly.body.foraging_body as fb
    rng = np.random.default_rng(0)
    rate = 1.3 * fb.DNG100_REST_HZ
    bursts = rng.poisson(2 * rate / 4 * 1e-3, 60000)                  # 2 cells, bursts of 4 spikes
    spikes = np.convolve(bursts, np.ones(12) / 3, "full")[:len(bursts)]  # ... over 12 ms, 1 ms bins
    hz = np.convolve(spikes, np.ones(50), "full")[:len(spikes)] / 2 / 0.05
    b = fb.ForagingBody(neural=True, seed=0, spontaneous_takeoff_per_s=0.0)      # as in Habitat
    v = []
    for k, h in enumerate(hz):
        b._walking = True
        b.update(1.0, {"hz_DNg100": float(h)}, float(k))
        v.append(min(b.state.speed_mm_s, 20.0))
    target = hz.mean() / fb.DNG100_REST_HZ * fb.WALK_SPEED_MM_S
    assert np.mean(v[5000:]) / target > 0.85
