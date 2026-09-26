"""
The fly exam's tests. Each test has a published (or calibration) target with
its citation, a tolerance, and a function that measures the model.

A test's run(ctx, quick) returns {"value": float, ...extras}; its band
(lo, hi) decides pass/fail, and the score decays linearly from 1 at the band
edge to 0 one band-width (or `scale`) outside it.

Groups:
  shiu      reproductions of Shiu et al. 2024 (Nature 634:210) predictions
  rest      resting activity with spontaneous ORN input vs recordings
  calib     our calibration targets (data/metadata/dynamics_calibrated.json)
  symmetry  left/right symmetry on mirrored stimuli
  health    activity health (no runaway / saturation)
"""
from __future__ import annotations

import numpy as np

from cognition.exam import core

SHIU = "Shiu et al. 2024, Nature 634:210"


def _ms(quick, q, f):
    return q if quick else f


def _seeds(quick, nq=2, nf=4):
    return tuple(range(1, (nq if quick else nf) + 1))


# ------------------------------------------------------------------- shiu
def sugar_mn9(ctx, quick):
    g = core.groups()
    ms = _ms(quick, 500, 1000)
    r = ctx.rates([(g["sugar"], 100.0)], ms, _seeds(quick))
    return {"value": float(r[g["mn9"]].mean())}


def sugar_dose(ctx, quick):
    g = core.groups()
    ms = _ms(quick, 400, 1000)
    vals = [float(ctx.rates([(g["sugar"], f)], ms, _seeds(quick))[g["mn9"]].mean()) for f in (10, 50, 200)]
    mono = float(vals[0] <= vals[1] + 1 and vals[1] <= vals[2] + 1)
    return {"value": mono, "mn9_hz_10_50_200": [round(v, 1) for v in vals]}


def bitter_mn9(ctx, quick):
    g = core.groups()
    r = ctx.rates([(g["bitter"], 100.0)], _ms(quick, 500, 1000), _seeds(quick))
    return {"value": float(r[g["mn9"]].mean())}


def sugar_bitter(ctx, quick):
    g = core.groups()
    ms, sd = _ms(quick, 500, 1000), _seeds(quick)
    a = float(ctx.rates([(g["sugar"], 100.0)], ms, sd)[g["mn9"]].mean())
    b = float(ctx.rates([(g["sugar"], 100.0), (g["bitter"], 200.0)], ms, sd)[g["mn9"]].mean())
    return {"value": b / a if a > 0 else float("nan"), "sugar_hz": round(a, 1), "sugar_bitter_hz": round(b, 1)}


def sugar_ir94e(ctx, quick):
    g = core.groups()
    ms, sd = _ms(quick, 500, 1000), _seeds(quick)
    a = float(ctx.rates([(g["sugar"], 100.0)], ms, sd)[g["mn9"]].mean())
    b = float(ctx.rates([(g["sugar"], 100.0), (g["ir94e"], 200.0)], ms, sd)[g["mn9"]].mean())
    return {"value": b / a if a > 0 else float("nan"), "sugar_hz": round(a, 1), "sugar_ir94e_hz": round(b, 1)}


def contralateral_mn9(ctx, quick):
    """All labelled sugar GRNs are left-side: MN9 right should fire more."""
    g = core.groups()
    r = ctx.rates([(g["sugar"], 100.0)], _ms(quick, 500, 1000), _seeds(quick))
    L, R = float(r[g["mn9_L"]].mean()), float(r[g["mn9_R"]].mean())
    return {"value": R - L, "mn9_left_hz": round(L, 1), "mn9_right_hz": round(R, 1)}


def sugar_responders(ctx, quick):
    g = core.groups()
    r = ctx.rates([(g["sugar"], 100.0)], _ms(quick, 500, 1000), _seeds(quick))
    resp = {k: float(r[v].mean()) for k, v in g["named"].items() if len(v)}
    return {"value": float(np.mean([v > 0 for v in resp.values()])),
            "rates_hz": {k: round(v, 1) for k, v in resp.items()}}


# Shiu et al. 2024 model predictions for 50 Hz activation of each named type:
# sufficient to activate MN9 (8 of 10 confirmed by optogenetics).
SUFFICIENT = {"Roundup": True, "Fdg": True, "FMIn": True, "Clavicle": True, "Fudog": True,
              "Rattle": True, "Zorro": True, "Bract": True, "Phantom": False, "Usnea": False}


def sufficiency(ctx, quick):
    g = core.groups()
    ms, sd = _ms(quick, 400, 1000), _seeds(quick)
    got = {}
    for k, pred in SUFFICIENT.items():
        cells = g["named"].get(k)
        if cells is None or not len(cells):
            continue
        got[k] = float(ctx.rates([(cells, 50.0)], ms, sd)[g["mn9"]].mean())
    agree = [((got[k] > 0) == SUFFICIENT[k]) for k in got]
    return {"value": float(np.mean(agree)), "mn9_hz": {k: round(v, 1) for k, v in got.items()}}


def necessity(ctx, quick):
    """Silencing the named pre-motor neurons changes sugar-evoked MN9 as the
    Shiu et al. 2024 model predicts (their criterion: required = MN9 <= 80% of
    control): Roundup required (Supp. Table 1C: 0.44-0.80), Fdg NOT required
    (0.85-0.93; Fdg's role in feeding is also contested experimentally:
    required in Flood et al. 2013 Nature 499:83, not in Shiu, Sterne et al.
    2022 eLife 11:e79887). Earlier versions of this test wrongly expected Fdg
    to be required; the published model fails that (0.93) as ours did (0.84)."""
    g = core.groups()
    ms, sd = _ms(quick, 500, 1000), _seeds(quick)
    base = float(ctx.rates([(g["sugar"], 100.0)], ms, sd)[g["mn9"]].mean())
    out = {}
    for k in ("Roundup", "Fdg"):
        cells = g["named"][k]
        acc = 0.0
        for s in sd:
            ctx.e.reset(seed=s)
            if ctx.lesioned.size:
                ctx.e.silence(ctx.lesioned)
            ctx.e.silence(cells)
            ctx.set_inputs([(g["sugar"], 100.0)])
            acc += float(ctx.window(ms)[g["mn9"]].mean())
        ctx.e.unsilence(cells)
        out[k] = acc / len(sd) / (ms / 1000.0) / base if base > 0 else float("nan")
    expect = {"Roundup": True, "Fdg": False}          # required? (Shiu et al. 2024 model)
    ok = [(out[k] <= 0.8) == req for k, req in expect.items()]
    return {"value": float(np.mean(ok)), "ratio": {k: round(v, 2) for k, v in out.items()}}


# ------------------------------------------------------------------- rest
def _rest_rates(ctx, quick):
    key = ("rest", quick)
    if key not in ctx.__dict__.setdefault("_cache", {}):
        ms = _ms(quick, 1500, 3000)
        ri, rr = core.resting()
        ctx._cache[key] = ctx.rates([(ri, rr)], ms, _seeds(quick, 1, 2), warm_inputs=[(ri, rr)], warm_ms=500)
    return ctx._cache[key]


def rest_group(name):
    def f(ctx, quick):
        r = _rest_rates(ctx, quick)
        return {"value": float(r[core.groups()[name]].mean())}
    return f


# ------------------------------------------------------------------ calib
def _mixture_sets(ctx, quick):
    from cognition.calibrate_dynamics import _mixtures
    c = core.connectome()
    n = c.neurons
    t = n["primary_type"].fillna("").astype(str).to_numpy()
    mixes = _mixtures(n, t, n_mix=4 if quick else 8, k=6, seed=7)
    g = core.groups()
    sets, latch = [], []
    for idx, rate, _ in mixes:
        o = np.argsort(idx)
        ctx.e.reset(seed=3)
        ctx.set_inputs([(idx[o], rate[o])])
        sc = ctx.window(500)
        sets.append(set(np.flatnonzero(sc[g["kc"]] > 0)))
        ctx.set_inputs([])
        ctx.window(200)
        after = ctx.window(300)
        latch.append(after.sum() / max(1, sc.sum()))
    return sets, latch


def kc_mixture_frac(ctx, quick):
    sets, latch = _mixture_sets(ctx, quick)
    ctx.__dict__.setdefault("_cache", {})["mix"] = (sets, latch)
    return {"value": float(np.mean([len(s) for s in sets]) / len(core.groups()["kc"]))}


def kc_mixture_overlap(ctx, quick):
    sets, _ = ctx.__dict__.get("_cache", {}).get("mix") or _mixture_sets(ctx, quick)
    import itertools
    jac = [len(a & b) / max(1, len(a | b)) for a, b in itertools.combinations(sets, 2)]
    return {"value": float(np.mean(jac))}


def no_latching(ctx, quick):
    _, latch = ctx.__dict__.get("_cache", {}).get("mix") or _mixture_sets(ctx, quick)
    return {"value": float(np.max(latch))}


def glomerulus_specificity(ctx, quick):
    c = core.connectome()
    n = c.neurons
    t = n["primary_type"].fillna("").astype(str).to_numpy()
    pn = n["class"].fillna("").astype(str).to_numpy() == "ALPN"
    vals = []
    for glom in (("DM1", "DL5") if quick else ("DM1", "DL5", "VA2", "DA1")):
        orn = np.flatnonzero(t == "ORN_" + glom)
        own = pn & np.char.startswith(t.astype(str), glom + "_")
        sc = ctx.trial([(orn, 60.0)], 500, seed=1)
        tot = sc[pn].sum()
        vals.append(float(sc[own].sum() / tot) if tot else 0.0)
    return {"value": float(np.mean(vals))}


OLSEN_RMAX, OLSEN_SIGMA = (144.0, 170.0), (11.8, 44.8)


def pn_transform(ctx, quick):
    """ORN -> PN input-output function of single glomeruli (DL5, VM7, DM1,
    DA1): ORNs of one type driven at 10-200 Hz above rest, PN rate above
    baseline over 500 ms, compared with the measured transform
    PN = Rmax ORN^1.5 / (ORN^1.5 + sigma^1.5) (lateral suppression s ~ 0 for a
    single glomerulus; Olsen, Bhandawat & Wilson 2010, Neuron 66:287; Rmax
    144-170 Hz, sigma 11.8-44.8 Hz). Value = fraction of (glomerulus, rate)
    points inside the band spanned by those parameter ranges, widened by 20%."""
    c = core.connectome()
    n = c.neurons
    t = n["primary_type"].fillna("").astype(str).to_numpy()
    pn = n["class"].fillna("").astype(str).to_numpy() == "ALPN"
    ri, rr = core.resting()
    ms = 500
    rates = (10.0, 50.0, 200.0) if quick else (10.0, 25.0, 50.0, 100.0, 200.0)
    gloms = ("DL5", "VM7") if quick else ("DL5", "VM7", "DM1", "DA1")

    def band(x):
        vals = [R * x ** 1.5 / (x ** 1.5 + sg ** 1.5) for R in OLSEN_RMAX for sg in OLSEN_SIGMA]
        return 0.8 * min(vals), 1.2 * max(vals)
    ok, detail = [], {}
    for glom in gloms:
        orn = np.flatnonzero(t == "ORN_" + glom)
        own = np.flatnonzero(pn & np.char.startswith(t.astype(str), glom + "_") & ~np.char.startswith(t.astype(str), glom + "_m"))
        if not len(orn) or not len(own):
            continue
        base = ctx.rates([(ri, rr)], ms, _seeds(quick, 1, 2))[own].mean()
        rest_orn = float(np.mean(rr[np.isin(ri, orn)])) if np.isin(ri, orn).any() else 0.0
        pts = []
        for x in rates:
            inp = [(ri, rr), (orn, rest_orn + x)]
            r = ctx.rates(inp, ms, _seeds(quick, 1, 2))[own].mean() - base
            lo, hi = band(x)
            ok.append(lo <= r <= hi)
            pts.append(round(float(r), 1))
        detail[glom] = pts
    return {"value": float(np.mean(ok)) if ok else float("nan"), "pn_minus_baseline_hz": detail,
            "orn_rates_above_rest_hz": list(rates)}


def loom_gf(ctx, quick):
    g = core.groups()
    r = ctx.rates([(g["loom"], 150.0)], 300, _seeds(quick))
    return {"value": float(r[g["gf"]].mean())}


def _with_rest(inputs):
    ri, rr = core.resting()
    return [(ri, rr)] + inputs


def odour_lateralization(ctx, quick):
    """Fermenting-fruit odour at one antenna: steering toward that side."""
    _, _, _, osp = core.orn_space()
    od = osp.sources["fermenting_fruit"] * 0.5
    z = np.zeros_like(od)
    ro = core.readout()
    out = {}
    for label, cl, cr in (("left", od, z), ("right", z, od)):
        acc = []
        for s in _seeds(quick):
            sc = ctx.trial(core.odour_inputs(cl, cr), 1500, s)
            acc.append(ro.channels(sc, 1500.0)["turn_bias"])
        out[label] = float(np.mean(acc))
    # right-antenna odour must steer further right than left-antenna odour
    return {"value": out["right"] - out["left"], "turn_bias": {k: round(v, 3) for k, v in out.items()}}


def _lc10a():
    if not hasattr(_lc10a, "rf"):
        from brain.sensory.retinotopy import load_retinotopy, receptive_fields_2hop
        rf = receptive_fields_2hop(load_retinotopy(core.connectome()), "LC10a").dropna(subset=["azimuth_deg"])
        _lc10a.rf = (rf["idx"].to_numpy(np.int64), rf.azimuth_deg.to_numpy(), rf.elevation_deg.to_numpy(),
                     np.clip(rf.rf_radius_deg.to_numpy(), 8, 40))
    return _lc10a.rf


def _target(az, size=13.0, hz=150.0):
    from simulation.stimuli.looming import angular_distance_deg
    idx, a, e, sig = _lc10a()
    d = angular_distance_deg(az, 0.0, a, e)
    return (idx, hz * np.exp(-np.maximum(0, d - size / 2) ** 2 / (2 * sig ** 2)))


def pursuit_sign(ctx, quick):
    ro = core.readout()
    out = {}
    for az in (-30.0, 30.0):
        acc = []
        for s in _seeds(quick):
            sc = ctx.trial(_with_rest([_target(az)]), 1500, s)
            acc.append(ro.channels(sc, 1500.0)["turn_bias"])
        out[az] = float(np.mean(acc))
    return {"value": out[30.0] - out[-30.0], "turn_bias_left_right": [round(out[-30.0], 3), round(out[30.0], 3)]}


def compass_tracking(ctx, quick):
    from brain.navigation.compass import Compass, CompassDrive
    drive = CompassDrive(Compass(core.connectome()))
    errs = []
    for h in ((45.0, 200.0) if quick else (0.0, 90.0, 180.0, 270.0)):
        drive.set_heading(h)
        ctx.e.reset(seed=3)
        ctx.set_inputs([(drive.indices, drive.rates_hz())])
        ctx.window(200)
        est, strength = drive.heading_estimate(ctx.window(300))
        errs.append(abs((est - h + 180) % 360 - 180) if est == est else 180.0)
    return {"value": float(np.max(errs))}


def goal_steering(ctx, quick):
    from brain.navigation.compass import Compass, CompassDrive
    from brain.navigation.goal import GoalCircuit, GoalDrive
    cx = Compass(core.connectome())
    comp, goal = CompassDrive(cx, 270.0), GoalDrive(GoalCircuit(core.connectome(), cx), peak_hz=120.0)
    res = {}
    for off in (-90.0, 90.0):
        goal.set_goal(270.0 + off)
        acc = []
        for s in _seeds(quick):
            sc = ctx.trial(_with_rest([(comp.indices, comp.rates_hz()), (goal.indices, goal.rates_hz())]),
                           600, s, warm_inputs=_with_rest([(comp.indices, comp.rates_hz()), (goal.indices, goal.rates_hz())]),
                           warm_ms=300)
            rd = goal.readout(sc, 600.0)
            acc.append(rd["DNa02_R"] - rd["DNa02_L"])
        res[off] = float(np.mean(acc))
    # goal clockwise (-90) must steer more to the right than counter-clockwise (+90)
    return {"value": res[-90.0] - res[90.0], "dna02_r_minus_l": {str(k): round(v, 1) for k, v in res.items()}}


def sound_startle(ctx, quick):
    """Sudden sound (JO-A/B) while food odour drives walking (DNg100 up):
    escape readiness (DNp11) rises and the walking command drops. DNg100
    fires only sporadically at rest, so the drop is measured from an
    odour-driven walking state."""
    g = core.groups()
    _, _, _, osp = core.orn_space()
    od = osp.sources["fermenting_fruit"] * 0.5
    walk = core.odour_inputs(od, od)
    sd, ms = _seeds(quick, 3, 5), 1000
    base = ctx.rates(walk, ms, sd, warm_inputs=walk, warm_ms=500)
    snd = ctx.rates(walk + [(g["jo_ab"], 150.0)], ms, sd, warm_inputs=walk, warm_ms=500)
    up = float(snd[g["dnp11"]].mean() - base[g["dnp11"]].mean())
    b100 = float(base[g["dng100"]].mean())
    down = float(1 - snd[g["dng100"]].mean() / b100) if b100 > 0 else float("nan")
    return {"value": float(up >= 20 and down >= 0.5), "dnp11_up_hz": round(up, 1),
            "dng100_walking_hz": round(b100, 1), "dng100_drop": round(down, 2) if down == down else None}


def mb_conditioning(ctx, quick):
    """Pair one odour with punishment (PPL101); the paired odour's MBON11
    response drops, the unpaired one's much less (Hige et al. 2015)."""
    from brain.plasticity.mushroom_body import MushroomBody
    from cognition.calibrate_dynamics import _mixtures
    c = ctx.c                                   # this configuration's wiring
    t = c.neurons["primary_type"].fillna("").astype(str).to_numpy()
    mixes = _mixtures(c.neurons, t, n_mix=3, k=6, seed=7)
    A = (np.sort(mixes[2][0]), mixes[2][1][np.argsort(mixes[2][0])])
    B = (np.sort(mixes[0][0]), mixes[0][1][np.argsort(mixes[0][0])])
    dan = np.flatnonzero(t == "PPL101")
    mbon11 = np.flatnonzero(t == "MBON11")
    saved = ctx.e.plastic_multipliers().copy()
    mb = MushroomBody(c, ctx.e, plastic=True, kc_mbon_gain=8.0, dan_modulatory=True)

    def present(od, seed, with_dan=False):
        ctx.e.reset(seed=seed); mb.reset_activity()
        ctx.set_inputs([od])
        tot = np.zeros(c.n)
        for ms in range(500):
            if with_dan and ms == 100:
                ctx.set_inputs([od, (dan, 100.0)])
            spk = ctx.e.run_collect(int(round(1.0 / ctx.e.p.dt)))
            mb.step(spk)
            np.add.at(tot, spk, 1)
        return tot[mbon11].sum()

    def test():
        mb.learning = False
        r = [np.mean([present(od, s) for s in (11, 12)]) for od in (A, B)]
        mb.learning = True
        return r
    before = test()
    for trial in range(2 if quick else 3):
        present(A, 100 + trial, True)
        present(B, 200 + trial, False)
    after = test()
    ctx.e.plastic_multipliers()[:] = saved
    ch = [(a - b) / max(b, 1e-9) for a, b in zip(after, before)]
    return {"value": ch[1] - ch[0], "paired_change": round(ch[0], 2), "unpaired_change": round(ch[1], 2),
            "mbon11_before": [round(x, 1) for x in before]}


# --------------------------------------------------------------- symmetry
def _sym(a, b):
    a, b = abs(a), abs(b)
    return 1.0 - abs(a - b) / (a + b) if a + b > 0 else float("nan")


def symmetry_loom(ctx, quick):
    """Looming from the left vs right: GF / escape responses should be mirror images."""
    from brain.sensory.encoders import LoomingEncoder
    from brain.sensory.retinotopy import load_retinotopy
    from simulation.stimuli.looming import LoomingStimulus
    enc = LoomingEncoder(core.connectome(), load_retinotopy(core.connectome()))
    g = core.groups()
    gfL, gfR = core.types("DNp01", "left"), core.types("DNp01", "right")
    res = {}
    for az in (-45.0, 45.0):
        st = LoomingStimulus(azimuth_deg=az, elevation_deg=0.0, half_size_mm=5.0, speed_mm_s=250.0,
                             start_distance_mm=50.0, t_start_ms=0.0)
        acc = np.zeros(core.connectome().n)
        for s in _seeds(quick):
            ctx.e.reset(seed=s)
            for k in range(20):
                idx, r = enc.indices, enc.rates_hz(k * 10.0, st)
                o = np.argsort(idx)
                ctx.set_inputs([(idx[o], r[o])])
                acc += ctx.window(10)
        res[az] = (float(acc[gfL].sum()), float(acc[gfR].sum()))
    ipsi = res[-45.0][0] + res[45.0][1]
    return {"value": _sym(res[-45.0][0] + res[-45.0][1], res[45.0][0] + res[45.0][1]),
            "gf_LR_for_left": res[-45.0], "gf_LR_for_right": res[45.0], "ipsi_total": ipsi}


def symmetry_pursuit(ctx, quick):
    r = pursuit_sign(ctx, quick)
    lb, rb = r["turn_bias_left_right"]
    ri, rr = core.resting()
    ro = core.readout()
    base = np.mean([ro.channels(ctx.trial([(ri, rr)], 1500, s), 1500.0)["turn_bias"] for s in _seeds(quick)])
    return {"value": _sym(lb - base, rb - base), "left_resp": round(lb - base, 3), "right_resp": round(rb - base, 3)}


def symmetry_odour(ctx, quick):
    r = odour_lateralization(ctx, quick)
    ro = core.readout()
    base = np.mean([ro.channels(ctx.trial(core.odour_inputs(*(2 * [np.zeros(len(core.orn_space()[3].odorants))])), 1500, s), 1500.0)["turn_bias"]
                    for s in _seeds(quick)])
    l, rgt = r["turn_bias"]["left"] - base, r["turn_bias"]["right"] - base
    return {"value": _sym(l, rgt), "left_resp": round(l, 3), "right_resp": round(rgt, 3)}


# ----------------------------------------------------------------- health
def health_saturation(ctx, quick):
    """Resting + sugar: fraction of non-sensory neurons firing > 250 Hz."""
    g = core.groups()
    ri, rr = core.resting()
    r = ctx.rates([(ri, rr), (g["sugar"], 100.0)], 500, _seeds(quick, 1, 2))
    ns = np.ones(core.connectome().n, bool)
    ns[g["sensory"]] = False
    return {"value": float((r[ns] > 250).mean()), "active_frac": round(float((r[ns] > 0).mean()), 4)}


def health_rate_cv(ctx, quick):
    """Population-rate variability (CV over 50 ms bins) at rest: runaway or bursting gives a high CV."""
    ri, rr = core.resting()
    ctx.e.reset(seed=5)
    ctx.set_inputs([(ri, rr)])
    ctx.window(300)
    bins = [ctx.window(50).sum() for _ in range(20 if quick else 40)]
    m = np.mean(bins)
    return {"value": float(np.std(bins) / m) if m > 0 else float("nan"), "mean_spikes_per_50ms": round(float(m), 1)}


# ------------------------------------------------------------ the catalog
def T(name, group, fn, lo, hi, target, cite, scale=None, core_=False, quick=True):
    return dict(name=name, group=group, fn=fn, lo=lo, hi=hi, target=target, cite=cite,
                scale=scale, core=core_, quick=quick)


INF = float("inf")
TESTS = [
    T("sugar_activates_MN9", "shiu", sugar_mn9, 5, INF, "sugar GRNs 100 Hz -> MN9 fires", SHIU, scale=20, core_=True),
    T("sugar_dose_monotonic", "shiu", sugar_dose, 1, 1, "MN9 rate rises with sugar GRN rate (10/50/200 Hz)", SHIU),
    T("bitter_no_MN9", "shiu", bitter_mn9, -INF, 1, "bitter GRNs -> MN9 silent", SHIU, scale=10, core_=True),
    T("bitter_inhibits_sugar", "shiu", sugar_bitter, -INF, 0.5, "sugar + strong bitter -> MN9 strongly reduced", SHIU, scale=0.5),
    T("ir94e_reduces_not_abolishes", "shiu", sugar_ir94e, 0.1, 0.99, "sugar + Ir94e -> MN9 reduced but not eliminated", SHIU, scale=0.3),
    T("contralateral_MN9", "shiu", contralateral_mn9, 0.01, INF, "unilateral sugar -> contralateral MN9 stronger", SHIU, scale=10),
    T("sugar_pathway_responders", "shiu", sugar_responders, 0.8, 1, "10 named 2nd-order/premotor types respond to sugar", SHIU, scale=0.5),
    T("named_sufficiency", "shiu", sufficiency, 0.8, 1, "50 Hz activation -> MN9 as the Shiu model predicts (8 yes, 2 no)", SHIU, scale=0.5),
    T("named_necessity", "shiu", necessity, 1, 1, "silencing: Roundup required (MN9 <= 80%), Fdg not (> 80%), as the Shiu model predicts", SHIU, scale=1),
    T("rest_uPN_Hz", "rest", rest_group("upn"), 0.5, 9.0, "uniglomerular PNs 4.6 +- 4.2 Hz",
      "Kazama & Wilson 2009 Nat Neurosci; Turner et al. 2008", scale=10),
    T("rest_LN_Hz", "rest", rest_group("ln"), 1.0, 10.0, "AL local neurons 4.6 +- 2.8 Hz", "Nagel, Hong & Wilson 2015 Nat Neurosci", scale=10),
    T("rest_KC_Hz", "rest", rest_group("kc"), 0.0, 0.5, "Kenyon cells 0.1 +- 0.4 Hz", "Turner, Bazhenov & Laurent 2008 J Neurophysiol", scale=2),
    T("rest_LHN_Hz", "rest", rest_group("lhn"), 0.0, 5.0, "lateral horn neurons low (spontaneous spikes suppressed)", "Jeanne & Wilson 2015 Neuron", scale=10),
    T("rest_GF_silent", "rest", rest_group("gf"), 0.0, 0.2, "Giant Fibre silent at rest", "Mu et al. 2014 J Exp Biol", scale=2, core_=True),
    T("rest_MBON_Hz", "rest", rest_group("mbon"), 0.3, 15.0, "MBONs tonically active (one ex vivo measurement: 12 Hz)", "Hafez et al. 2023 eLife", scale=15),
    T("kc_mixture_sparseness", "calib", kc_mixture_frac, 0.02, 0.10, "~5-10% of KCs per odour", "Turner et al. 2008; Honegger et al. 2011", scale=0.1),
    T("kc_mixture_overlap", "calib", kc_mixture_overlap, 0.0, 0.15, "distinct KC codes for different odours", "Lin et al. 2014 Nat Neurosci", scale=0.3),
    T("no_latching", "calib", no_latching, 0.0, 0.05, "activity decays after odour offset", "calibration (dynamics_calibrated.json)", scale=0.3),
    T("pn_odour_transform", "calib", pn_transform, 0.7, 1.0,
      "ORN->PN transform: PN rise vs ORN rate matches the measured curve (Rmax 144-170 Hz, sigma 12-45 Hz)",
      "Olsen, Bhandawat & Wilson 2010 Neuron 66:287", scale=0.7),
    T("glomerulus_specificity", "calib", glomerulus_specificity, 0.5, 1.0, "single receptor type drives mostly its own PNs", "calibration; Bhandawat et al. 2007", scale=0.5),
    T("looming_drives_GF", "calib", loom_gf, 100, INF, "LC4/LPLC2 150 Hz -> Giant Fibre", "von Reyn et al. 2014; Shiu et al. 2024", scale=150, core_=True),
    T("odour_lateralization", "calib", odour_lateralization, 0.01, INF, "odour at right antenna steers right vs left", "Borst & Heisenberg 1982; Gaudry et al. 2013", scale=0.05),
    T("pursuit_sign", "calib", pursuit_sign, 0.02, INF, "LC10a target right steers right vs left", "Hindmarsh Sten et al. 2021; Collie et al. 2026", scale=0.1, core_=True),
    T("compass_tracking_deg", "calib", compass_tracking, 0.0, 30.0, "E-PG bump reads back the heading", "Seelig & Jayaraman 2015", scale=60),
    T("goal_steering_sign", "calib", goal_steering, 5.0, INF, "goal clockwise -> DNa02 right > left", "Westeinde et al. 2024; Mussells Pires et al. 2024", scale=20),
    T("sound_startle", "calib", sound_startle, 1, 1, "JO-A/B burst -> DNp11 up, walking command down", "calibration (hearing); Lehnert et al. 2013", scale=1),
    T("mb_conditioning", "calib", mb_conditioning, 0.3, INF, "paired odour's MBON11 drops more than unpaired", "Hige et al. 2015 Neuron", scale=0.5, quick=False),
    T("symmetry_looming", "symmetry", symmetry_loom, 0.8, 1.0, "mirror-image GF responses to left/right looming", "bilateral symmetry", scale=0.5),
    T("symmetry_pursuit", "symmetry", symmetry_pursuit, 0.6, 1.0, "left/right pursuit responses of similar size", "bilateral symmetry", scale=0.6),
    T("symmetry_odour", "symmetry", symmetry_odour, 0.6, 1.0, "left/right odour steering of similar size", "bilateral symmetry", scale=0.6),
    T("health_saturation", "health", health_saturation, 0.0, 0.001, "almost no neurons pinned >250 Hz", "activity health", scale=0.01),
    T("health_rate_cv", "health", health_rate_cv, 0.0, 0.5, "steady population rate at rest", "activity health", scale=1.0),
]
BY_NAME = {t["name"]: t for t in TESTS}
CORE = [t["name"] for t in TESTS if t["core"]]


def score(test: dict, value: float) -> tuple:
    lo, hi = test["lo"], test["hi"]
    if value != value:                        # NaN
        return False, 0.0
    if lo <= value <= hi:
        return True, 1.0
    width = test["scale"] or (hi - lo if np.isfinite(hi - lo) and hi > lo else 1.0)
    dist = (lo - value) if value < lo else (value - hi)
    return False, float(max(0.0, 1.0 - dist / width))
