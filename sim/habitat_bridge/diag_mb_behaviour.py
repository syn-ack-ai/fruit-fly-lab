"""
Can mushroom-body learning change what the pet DOES? (diagnostic)

1. Structure: how much of the input to the walking/steering descending neurons
   (DNg100, DNp09, DNa01, DNa02, MDN) comes from MBONs within one and two
   synapses (share of excitatory synapse counts, multiplied along paths).
2. Function: present the bowl odour (fermenting fruit, as in home.py) at a
   concentration seen ~1 m from the bowl, left-biased, before and after
   conditioning (bowl odour paired with reward dopamine, PAM05-08 at 100 Hz, as
   when the pet eats at the bowl), and optionally with weights learned in a
   Habitat lifetime (--weights). Report MBON rates and the motor channels.

    PYTHONPATH=. FLY_DYNAMICS=calibrated .venv/bin/python -m sim.habitat_bridge.diag_mb_behaviour [--weights on/mb_weights.npy]
"""
from __future__ import annotations

import argparse
import collections

import numpy as np

MOTOR = ("DNg100", "DNp09", "DNa01", "DNa02", "MDN", "DNge078")


def structure(c):
    n = c.neurons
    t = n["primary_type"].fillna("").astype(str).to_numpy()
    mbon = np.char.startswith(t.astype(str), "MBON")
    W = c.w.tocsr()
    WT = W.T.tocsr()
    out = {}
    tot_in = np.asarray(abs(WT).sum(axis=1)).ravel()                 # all input synapses per neuron
    frac = W.multiply(1.0 / np.maximum(tot_in, 1)[None, :]).tocsr()   # pre->post share of post's input
    m = np.zeros(c.n); m[mbon] = 1.0
    one = frac.T @ m                                                  # share of input from MBONs directly
    two = frac.T @ one                                                # via one intermediate neuron
    for ty in MOTOR:
        idx = np.flatnonzero(t == ty)
        out[ty] = {"direct": round(float(one[idx].mean()), 4), "two_hop": round(float(two[idx].mean()), 4)}
    # which MBONs have the strongest 2-hop path to DNa02 / DNg100
    top = {}
    for ty in ("DNa02", "DNg100"):
        tgt = np.flatnonzero(t == ty)
        into = np.asarray(frac[:, tgt].sum(axis=1)).ravel()         # each neuron's share into the target
        path = frac @ into                                            # ... via one intermediate neuron
        cand = np.flatnonzero(mbon)
        o = cand[np.argsort(-np.abs(path[cand]))[:5]]
        top[ty] = [(t[i], n["side"].iloc[i][:1], round(float(path[i]), 4)) for i in o]
    return out, top


def function(c, weights_file=None, trials=6, seed=1):
    from brain.plasticity.mushroom_body import MushroomBody
    from brain.motor.descending import DescendingReadout
    from native.lif_native import NativeLIFEngine
    from simulation.engine.session import apply_dynamics
    from sim.habitat_bridge.home import HomeSenses, HomeWorld
    n = c.neurons
    t = n["primary_type"].fillna("").astype(str).to_numpy()
    e = NativeLIFEngine.from_connectome(c, seed=seed)
    apply_dynamics(e, c, "calibrated")
    mb = MushroomBody(c, e, plastic=True, kc_mbon_gain=8.0, dan_modulatory=True)
    ro = DescendingReadout(c)
    hw = HomeWorld(seed=seed)
    hs = HomeSenses(c, hw)
    mbon_types = t[mb.mbon]

    def present(conc_l, conc_r, reward=False, ms=1000, s=1, learn=False):
        hw.conc["L"] = np.array([conc_l] + [0.0] * (len(hw.sources) - 1))
        hw.conc["R"] = np.array([conc_r] + [0.0] * (len(hw.sources) - 1))
        hw.taste = "sweet" if reward else None
        idx = hs.indices
        rates = hs.rates_hz(0.0)
        e.reset(seed=s); mb.reset_activity(); mb.learning = learn
        e.set_poisson(idx, rates)
        counts = np.zeros(c.n)
        for _ in range(ms):
            spk = e.run_collect(int(round(1.0 / e.p.dt)))
            mb.step(spk, 1.0)
        counts = e.spike_counts.astype(float) / (ms / 1000.0)
        ch = ro.channels(e.spike_counts, float(ms))
        return counts, ch

    def test(label):
        rows = []
        for s in (11, 12, 13):
            cnt, ch = present(0.45, 0.30, s=s)
            rows.append((cnt, ch))
        cnt = np.mean([r[0] for r in rows], axis=0)
        ch = {k: float(np.mean([r[1].get(k, 0.0) for r in rows])) for k in rows[0][1]}
        mb_rates = collections.defaultdict(list)
        for i, mt in zip(mb.mbon, mbon_types):
            mb_rates[mt].append(cnt[i])
        dn = {ty: round(float(cnt[t == ty].mean()), 1) for ty in MOTOR}
        print(f"{label:28s} turn_bias {ch.get('turn_bias', 0):+.3f} fwd {ch.get('forward_walk', 0):.3f} "
              f"| DN Hz {dn} | depressed synapses {mb.summary()['depressed_synapses']}")
        return {k: float(np.mean(v)) for k, v in mb_rates.items()}, dn, ch

    before, dn0, ch0 = test("naive")
    if weights_file:
        w = np.load(weights_file)
        mb.weights[:] = w
        mb.mult[mb.edge_pos] = (mb.kc_mbon_gain * w).astype(np.float32)
        getattr(e, "commit_plastic", lambda *_: None)(mb.edge_pos)
        test("lifetime weights")
        mb.weights[:] = 1.0
        mb.mult[mb.edge_pos] = np.float32(mb.kc_mbon_gain)
        getattr(e, "commit_plastic", lambda *_: None)(mb.edge_pos)
    for k in range(trials):          # conditioning: bowl odour + reward dopamine (eating)
        present(0.9, 0.9, reward=True, ms=1000, s=100 + k, learn=True)
    after, dn1, ch1 = test("after %d rewarded pairings" % trials)
    changed = sorted(((mt, before[mt], after[mt]) for mt in before if abs(after[mt] - before[mt]) > 1.0),
                     key=lambda r: r[2] - r[1])
    print("MBONs whose odour response changed (Hz before -> after):",
          [(mt, round(b, 1), round(a, 1)) for mt, b, a in changed[:10]])
    print("motor change after conditioning: turn_bias %+.3f, forward %+.3f, DN Hz delta %s" % (
        ch1.get("turn_bias", 0) - ch0.get("turn_bias", 0), ch1.get("forward_walk", 0) - ch0.get("forward_walk", 0),
        {k: round(dn1[k] - dn0[k], 1) for k in dn0}))


def main():
    from brain.neurons.registry import load_connectome
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None)
    ap.add_argument("--trials", type=int, default=6)
    a = ap.parse_args()
    c = load_connectome()
    s, top = structure(c)
    print("share of each motor DN type's input from MBONs (direct / via one neuron):", s)
    print("strongest MBON two-synapse paths:", top)
    function(c, a.weights, a.trials)


if __name__ == "__main__":
    main()
