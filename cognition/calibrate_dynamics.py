"""
Calibrate the optional adaptation and slow-inhibition dynamics
(NativeLIFEngine.set_dynamics) against known biology.

The published model (Shiu et al. 2024) has one fast synapse type and no
adaptation. On the FlyWire connectome that lets excitation spread: a single
olfactory receptor type recruits ~70% of antennal-lobe projection neurons
within 25-100 ms, ~45% of Kenyon cells respond to any odour (largely the same
cells), and strong stimuli latch the brain into self-sustained activity.
Real flies keep projection-neuron responses glomerulus-specific and Kenyon
cell codes sparse (~5-10%), and activity decays after a stimulus, helped by
slow GABA-B inhibition and spike-frequency adaptation.

For each parameter set this measures, on FlyWire FAFB:
  pn_off     fraction of projection neurons OUTSIDE the stimulated glomerulus
             that fire (single receptor type at 60 Hz, 500 ms; mean of 4 odours)
  pn_own     fraction of projection-neuron spikes from the stimulated glomerulus
  kc_frac    fraction of Kenyon cells firing, and kc_overlap between odours
  latch      spikes 200-500 ms after the odours end, as a fraction of during
  mix_kc_frac, mix_kc_overlap   the same for 8 odour-like mixtures: each drives
             6 random olfactory receptor types (of ~50 glomeruli, as real odours
             activate several; e.g. Hallem & Carlson 2006) at 30-80 Hz.
             Target: ~5-10% of Kenyon cells, low overlap (Turner et al. 2008;
             Honegger et al. 2011)
  mn9_sugar, mn9_bitter, gf_loom   firing rates (Hz) that must be preserved:
             sugar -> MN9 fires, bitter -> MN9 silent, looming -> Giant Fibre
  rest       spikes with no input at all (must be 0)

  python -m cognition.calibrate_dynamics --grid default --workers 14 --out dyn_grid.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from multiprocessing import get_context

import numpy as np

ODORS = ["ORN_DM1", "ORN_DL5", "ORN_VA2", "ORN_DA1"]
N_MIX, MIX_GLOMERULI, MIX_SEED = 8, 6, 2024
_ctx = {}


def _edge_positions(c, pre_mask, post_mask):
    """CSR positions (engine order) of connections from pre_mask to post_mask."""
    w = c.w.tocsr(); pos = []
    for i in np.flatnonzero(pre_mask):
        a, b = w.indptr[i], w.indptr[i + 1]
        pos.append(a + np.flatnonzero(post_mask[w.indices[a:b]]))
    return np.concatenate(pos) if pos else np.empty(0, np.int64)


def _mixtures(n, t, n_mix=N_MIX, k=MIX_GLOMERULI, seed=MIX_SEED):
    """Odour-like stimuli: k random olfactory receptor types, each at 30-80 Hz."""
    # VP glomeruli are thermo-/hygrosensory, not olfactory
    types = sorted(s for s in set(t) if s.startswith("ORN_") and not s.startswith("ORN_VP"))
    rng = np.random.default_rng(seed); out = []
    for _ in range(n_mix):
        pick = rng.choice(len(types), k, replace=False)
        idx, rate = [], []
        for j in pick:
            ii = n[t == types[j]]["idx"].to_numpy(); r = rng.uniform(30.0, 80.0)
            idx.append(ii); rate.append(np.full(len(ii), r))
        out.append((np.concatenate(idx), np.concatenate(rate), [types[j] for j in pick]))
    return out


def _setup():
    if _ctx:
        return _ctx
    os.environ.setdefault("FLY_THREADS", "1")
    from brain.neurons.registry import load_connectome
    from brain.sensory.modalities import BY_KEY, resolve_neurons
    from native.lif_native import NativeLIFEngine
    c = load_connectome(); n = c.neurons
    t = n["primary_type"].fillna("").astype(str)
    cl = n["class"].fillna("").astype(str)
    pn = (cl == "ALPN").to_numpy()
    _ctx.update(
        c=c, t=t.to_numpy(), pn=pn,
        kc=n[t.str.match("^KC")]["idx"].to_numpy(),
        odors={o: np.sort(n[t == o]["idx"].to_numpy()) for o in ODORS},
        own={o: (pn & t.str.startswith(o[4:] + "_").to_numpy()) for o in ODORS},
        mixtures=_mixtures(n, t),
        mn9=n[t == "CB0701"]["idx"].to_numpy(),
        gf=n[t == "DNp01"]["idx"].to_numpy(),
        sugar=np.sort(resolve_neurons(BY_KEY["taste_sugar"], c)),
        bitter=np.sort(resolve_neurons(BY_KEY["taste_bitter"], c)),
        loom=np.sort(c.by_cell_types(["LC4", "LPLC2"])["idx"].to_numpy()),
        engine=NativeLIFEngine.from_connectome(c, seed=1, threads=1),
        # scopes for the extra dynamics
        pn_kc_pos=_edge_positions(c, (cl == "ALPN").to_numpy(), t.str.match("^KC").to_numpy()),
        eln_pn_pos=_edge_positions(c, ((cl == "ALLN") & (n["sign"] > 0)).to_numpy(), (cl == "ALPN").to_numpy()),
        eln_eln_pos=_edge_positions(c, ((cl == "ALLN") & (n["sign"] > 0)).to_numpy(),
                                    ((cl == "ALLN") & (n["sign"] > 0)).to_numpy()),
        scope={
            "all": np.ones(c.n, bool),
            "none": np.zeros(c.n, bool),
            "all_inhib": (n["sign"] < 0).to_numpy(),
            "al_ln_inhib": ((cl == "ALLN") & (n["sign"] < 0)).to_numpy(),
            "al_ln": (cl == "ALLN").to_numpy(),
        },
    )
    return _ctx


def _dynamics(params: dict, x: dict) -> dict:
    """params -> set_dynamics kwargs; *_scope picks which neurons get it."""
    p = dict(params)
    a_scope = x["scope"][p.pop("adapt_scope", "all")]
    s_scope = x["scope"][p.pop("slow_scope", "all_inhib")]
    ln_extra = p.pop("ln_adapt_mV", 0.0)
    p["adapt_mV"] = (np.where(a_scope, p.get("adapt_mV", 0.0), 0.0)
                     + np.where(x["scope"]["al_ln"], ln_extra, 0.0)).astype(np.float32)
    p["slow_ratio"] = np.where(s_scope, p.get("slow_ratio", 0.0), 0.0).astype(np.float32)
    return p


def _single_pn_share(x, sc):
    """Median share of a firing KC's PN drive that comes from its strongest PN."""
    if "WT" not in x:
        x["WT"] = abs(x["c"].w.T.tocsr())
    WT, pn = x["WT"], x["pn"]
    shares = []
    for k in x["kc"][sc[x["kc"]] > 0][:300]:
        row = WT.getrow(k); m = pn[row.indices]
        d = row.data[m] * sc[row.indices[m]]
        if d.sum() > 0:
            shares.append(d.max() / d.sum())
    return float(np.median(shares)) if shares else float("nan")


def evaluate(params: dict, seed: int = 1) -> dict:
    x = _setup(); e = x["engine"]
    params = dict(params)
    apl_gain = params.pop("apl_gain", 0.0)
    pn_kc_gain = params.pop("pn_kc_gain", 1.0)
    eln_pn_gain = params.pop("eln_pn_gain", 1.0)
    eln_eln_gain = params.pop("eln_eln_gain", 1.0)
    mult = e.plastic_multipliers()
    mult[:] = 1.0
    mult[x["pn_kc_pos"]] = np.float32(pn_kc_gain)
    mult[x["eln_pn_pos"]] = np.float32(eln_pn_gain)
    mult[x["eln_eln_pos"]] = np.float32(eln_eln_gain)
    gf_scale = params.pop("gf_scale", 0.2)
    if gf_scale != 1.0:                            # Giant Fibre excitability (see session.py)
        from simulation.engine.session import giant_fibre_multipliers
        key = ("gf", gf_scale)
        if key not in x:
            x[key] = giant_fibre_multipliers(x["c"], {"chemical_input_scale": gf_scale,
                                                     "johnstons_organ_input": 0.0})
        for pos, m in x[key]:
            mult[pos] *= np.float32(m)
    if params.pop("orn_pn_comp", True):            # Tobin et al. 2017 (see session.py)
        if "comp" not in x:
            from simulation.engine.session import orn_pn_compensation
            x["comp"] = orn_pn_compensation(x["c"])
        for pos, m in x["comp"]:
            mult[pos] *= np.float32(m)
    e.set_dynamics(**_dynamics(params, x))
    e.unsilence_all()
    mb = None
    if apl_gain > 0:
        from brain.plasticity.mushroom_body import MushroomBody
        mb = x.get("mb") or MushroomBody(x["c"], e, apl_gain=apl_gain, plastic=False)
        x["mb"] = mb; mb.apl_gain = apl_gain; e.silence(mb.apl)

    def run(ms):
        if mb is None:
            e.run(ms); return
        for _ in range(int(ms)):
            mb.step(e.run_collect(10))

    def reset():
        e.reset(seed=seed)
        if mb is not None:
            mb.reset_activity(); e.silence(mb.apl)
    out = {}
    # at rest
    reset(); run(200.0); out["rest"] = int(e.spike_counts.sum())
    # resting sensory activity (every ORN at its Hallem & Carlson 2006 rate): GF must stay silent
    if "orn_spont" not in x:
        from brain.sensory.olfaction import OlfactorySpace
        orn = np.flatnonzero(np.char.startswith(x["t"].astype(str), "ORN_"))
        types = sorted(set(x["t"][orn]))
        osp = OlfactorySpace([ty[4:] for ty in types]); gi = {ty: i for i, ty in enumerate(types)}
        sp = osp.spont
        x["orn_spont"] = (orn, np.array([sp[gi[x["t"][i]]] for i in orn]))
    reset(); e.set_poisson(*x["orn_spont"]); run(2000.0)
    out["gf_rest_hz"] = float(e.spike_counts[x["gf"]].mean() / 2.0)
    out["kc_rest_hz"] = float(e.spike_counts[x["kc"]].mean() / 2.0)
    e.clear_poisson()
    pn_off, pn_own, kc_sets, latch = [], [], [], []
    for o, idx in x["odors"].items():
        reset(); e.set_poisson(idx, 60.0); run(500.0)
        sc = e.spike_counts.copy()
        own = x["own"][o]; other = x["pn"] & ~own
        pn_off.append(float((sc[other] > 0).mean()))
        tot = sc[x["pn"]].sum()
        pn_own.append(float(sc[own].sum() / tot) if tot else 0.0)
        kc_sets.append(set(np.flatnonzero(sc[x["kc"]] > 0)))
        if o == ODORS[0]:
            out["kc_single_pn_share"] = _single_pn_share(x, sc)
        during = sc.sum()
        e.clear_poisson(); run(200.0); before = e.spike_counts.sum(); run(300.0)
        latch.append(float((e.spike_counts.sum() - before) / max(during, 1)))
    out["pn_off"] = float(np.mean(pn_off)); out["pn_own"] = float(np.mean(pn_own))
    out["kc_frac"] = float(np.mean([len(s) for s in kc_sets]) / len(x["kc"]))
    jac = [len(a & b) / max(1, len(a | b)) for a, b in itertools.combinations(kc_sets, 2)]
    out["kc_overlap"] = float(np.mean(jac))
    out["latch"] = float(np.mean(latch)); out["latch_max"] = float(np.max(latch))
    mix_sets, mix_latch = [], []
    for idx, rate, _ in x["mixtures"]:
        reset(); e.set_poisson(idx, rate); run(500.0)
        sc = e.spike_counts.copy()
        mix_sets.append(set(np.flatnonzero(sc[x["kc"]] > 0)))
        e.clear_poisson(); run(200.0); before = e.spike_counts.sum(); run(300.0)
        mix_latch.append(int(e.spike_counts.sum() - before))
    out["mix_kc_frac"] = float(np.mean([len(s) for s in mix_sets]) / len(x["kc"]))
    out["mix_kc_frac_range"] = [float(min(len(s) for s in mix_sets) / len(x["kc"])),
                                float(max(len(s) for s in mix_sets) / len(x["kc"]))]
    jac = [len(a & b) / max(1, len(a | b)) for a, b in itertools.combinations(mix_sets, 2)]
    out["mix_kc_overlap"] = float(np.mean(jac))
    out["mix_latch_spikes"] = int(max(mix_latch))
    for key, idx, rate, ms, read in (("mn9_sugar", "sugar", 150.0, 500.0, "mn9"),
                                     ("mn9_bitter", "bitter", 150.0, 500.0, "mn9"),
                                     ("gf_loom", "loom", 150.0, 300.0, "gf")):
        reset(); e.set_poisson(x[idx], rate); run(ms)
        r = x[read]
        out[key] = float(e.spike_counts[r].sum() / len(r) / (ms / 1000.0))
    return out


def _job(params):
    t0 = time.time()
    try:
        m = evaluate(params)
    except Exception as ex:  # keep the sweep going
        m = {"error": repr(ex)}
    return {"params": params, "metrics": m, "seconds": round(time.time() - t0, 1)}


GRIDS = {
    "default": dict(tau_adapt_ms=[100.0, 300.0], adapt_mV=[0.0, 1.0, 3.0, 8.0],
                    tau_slow_ms=[50.0, 200.0], slow_ratio=[0.0, 1.0, 3.0, 8.0]),
    # slow inhibition only from antennal-lobe inhibitory local neurons
    "adapt_fine": dict(slow_scope=["al_ln_inhib"], adapt_scope=["all"],
                       tau_adapt_ms=[1000.0, 2000.0], adapt_mV=[0.1, 0.2, 0.3, 0.5],
                       tau_slow_ms=[300.0], slow_ratio=[0.4, 0.5]),
    "pn_kc": dict(slow_scope=["al_ln_inhib"], adapt_scope=["all"], tau_adapt_ms=[1000.0], adapt_mV=[0.2],
                  tau_slow_ms=[300.0], slow_ratio=[0.1, 0.2, 0.3, 0.5],
                  pn_kc_gain=[1.0, 0.6, 0.4, 0.3, 0.2]),
    "eln": dict(slow_scope=["al_ln_inhib"], adapt_scope=["all"], tau_adapt_ms=[1000.0], adapt_mV=[0.2],
                tau_slow_ms=[300.0], slow_ratio=[0.0, 0.2, 0.5],
                eln_pn_gain=[1.0, 0.5, 0.25, 0.1, 0.0]),
    "eln_mix": dict(slow_scope=["al_ln_inhib"], adapt_scope=["all"], tau_adapt_ms=[1000.0], adapt_mV=[0.2, 0.5],
                    tau_slow_ms=[300.0], slow_ratio=[0.0, 0.2, 0.5],
                    eln_pn_gain=[1.0, 0.25, 0.1, 0.0]),
    "eln2": dict(slow_scope=["al_ln_inhib"], adapt_scope=["all"], tau_adapt_ms=[1000.0], adapt_mV=[0.2],
                 tau_slow_ms=[300.0], slow_ratio=[0.0, 0.1, 0.2],
                 eln_pn_gain=[0.0, 0.1, 0.2, 0.3], eln_eln_gain=[1.0, 0.5, 0.25], ln_adapt_mV=[0.0, 1.0]),
    "refine": dict(slow_scope=["al_ln_inhib"], adapt_scope=["all"],
                   tau_adapt_ms=[1000.0], adapt_mV=[0.5],
                   tau_slow_ms=[300.0, 600.0], slow_ratio=[0.3, 0.4, 0.5, 0.6, 0.8],
                   apl_gain=[0.0, 1.0, 4.0]),
    "al_slow": dict(slow_scope=["al_ln_inhib"], adapt_scope=["all"],
                    tau_adapt_ms=[300.0, 1000.0], adapt_mV=[0.0, 0.5, 1.0, 2.0],
                    tau_slow_ms=[100.0, 300.0], slow_ratio=[0.1, 0.3, 1.0, 3.0]),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--grid", default="default")
    ap.add_argument("--params", help="one JSON dict instead of a grid")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="dyn_grid.json")
    a = ap.parse_args()
    if a.params:
        print(json.dumps(_job(json.loads(a.params)), indent=1)); return
    g = GRIDS[a.grid]
    combos = [dict(zip(g, v)) for v in itertools.product(*g.values())]
    # published model once, and skip duplicate "off" settings
    seen, jobs = set(), []
    for p in combos:
        key = (p["adapt_mV"] > 0 and p["tau_adapt_ms"], p["adapt_mV"],
               p["slow_ratio"] > 0 and p["tau_slow_ms"], p["slow_ratio"],
               p.get("slow_scope"), p.get("adapt_scope"), p.get("apl_gain"), p.get("pn_kc_gain"), p.get("eln_pn_gain"),
               p.get("eln_eln_gain"), p.get("ln_adapt_mV"))
        if key not in seen:
            seen.add(key); jobs.append(p)
    print(f"{len(jobs)} parameter sets on {a.workers} workers", flush=True)
    res = []
    with get_context("spawn").Pool(a.workers) as pool:
        for r in pool.imap_unordered(_job, jobs):
            res.append(r)
            m, p = r["metrics"], r["params"]
            if "error" in m:
                print("ERROR", p, m["error"], flush=True); continue
            print(f"a={p['adapt_mV']:g}mV/{p['tau_adapt_ms']:g}ms s={p['slow_ratio']:g}/{p['tau_slow_ms']:g}ms "
                  f"pnkc={p.get('pn_kc_gain', 1):g} eln={p.get('eln_pn_gain', 1):g}/{p.get('eln_eln_gain', 1):g} lnA={p.get('ln_adapt_mV', 0):g} | 1PN {m.get('kc_single_pn_share', float('nan')):.2f} | "
                  f"PN off {m['pn_off']:.2f} own {m['pn_own']:.2f} | KC {m['kc_frac']:.3f} ov {m['kc_overlap']:.2f} | "
                  f"latch {m['latch']:.2f}/{m['latch_max']:.2f} | MIX KC {m['mix_kc_frac']:.3f} "
                  f"[{m['mix_kc_frac_range'][0]:.3f}-{m['mix_kc_frac_range'][1]:.3f}] ov {m['mix_kc_overlap']:.2f} "
                  f"latch {m['mix_latch_spikes']} | MN9 sugar {m['mn9_sugar']:.0f} bitter {m['mn9_bitter']:.0f} | "
                  f"GF {m['gf_loom']:.0f} | rest {m['rest']} ({r['seconds']}s)", flush=True)
            json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
