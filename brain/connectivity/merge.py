"""
The "complete map": the Janelia MaleCNS (brain + nerve cord of one male) with
its demonstrated reconstruction gaps filled. Selected with FLY_DATASET=merged.
FlyWire FAFB (the brain of one female) is the reference that decides what
counts as a gap. Every change is logged (data/derived/merged/fill_log.csv.gz).

What the data support (measured 2026-09-27, sim scratch analysis; numbers in
the build manifest):
  - Connection groups ((pre type, side) -> (post type, side), synapses per
    postsynaptic cell) are as left/right symmetric in the MaleCNS as in FAFB at
    every strength: male asymmetry beyond FAFB's 99th percentile 0.1-1%, and
    one-sided groups at the same rate. So a general "fill the weaker side" rule
    would mostly overwrite natural variation.
  - Groups strongly lopsided in the male (> 4x, stronger side >= 5 per cell)
    where FAFB is symmetric: 3,640; the reverse (FAFB lopsided where the male is
    symmetric): 4,135. Outside the olfactory receptor neurons the two animals
    are equally lopsided -- individual variation, not gaps.
  - Male vs female per type pair (both sides, after the overall 1.24x synapse
    scale): the male is > 4x below the female on 37,864 pairs and the female
    > 4x below the male on 77,044. Differences run both ways at large scale
    (individual, sex, typing and dataset differences), so a female fill would
    mostly import FAFB's own idiosyncrasies. Not done.
  - Rule 2 below covers families short on BOTH sides (JO, head bristles).
  - One gap is demonstrated: the male's LEFT ANTENNA olfactory receptor
    neurons. By wiring (brain/sensory/orn_side.py) 994 left / 1,637 right
    (FAFB 1,117 / 1,132); by type e.g. ORN_VA1v 19 / 111 (FAFB 51 / 43), VM4
    13 / 65, VA1d 25 / 107. (Johnston's organ looked like a gap by subtype --
    JO-A 0 / 24 -- but the family totals are 349 / 324: a typing difference,
    the right side's JO-A are mostly typed "JO-A-unclear".)

Rule (population gap): a cell type whose FAMILY (all types with the same
prefix: ORN_, JO-, BM_ ...) is lopsided in the male (|log2 left/right| >
FAMILY_GAP) but symmetric in FAFB (< FAFB_FAMILY_SYM), whose own neuron count
is lopsided the same way beyond FAFB's normal type-level variation (|log2| >
COUNT_GAP_IN_FAMILY = 0.6; FAFB's ORN types: 95th percentile 0.51, 99th 0.58)
while FAFB has the type symmetric (< FAFB_SYM), and which is not male-specific
/ dimorphic. (A family-level gap is the evidence; the first version required
|log2| > 1.0 per type as well and filled 12 ORN types, leaving the ORN->PN
input 5.3% lopsided.) Fill: every output synapse of that
type's neurons on the deficient side is scaled by n_other / n_deficient (at
most MAX_FILL), so each downstream cell receives the input the full population
would give (population compensation; the missing neurons themselves cannot be
recreated). Sides of olfactory receptor neurons come from their wiring.

C. APPROXIMATIONS: fewer, stronger synapses stand in for the missing neurons
(the same total drive with less independent noise); the deficient side's
remaining neurons are assumed representative of the missing ones.

    python -m brain.connectivity.merge          (writes data/derived/merged/)
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import scipy.sparse as sp

import config

COUNT_GAP_IN_FAMILY = 0.6   # ... for a type in a family that is itself a gap (FAFB ORN types: p99 0.58)
FAMILY_GAP = 0.5         # ... and of its family
FAFB_SYM = 0.5           # FAFB must be symmetric for the type
FAFB_FAMILY_SYM = 0.2    # ... and for the family
MIN_CELLS = 6            # FAFB neurons of the type (both sides) for a reliable count
FAMILY_MIN_TYPES = 5     # a "family" of one or two types is no evidence beyond the type itself
GROUP_GAP = 1.2          # in a gap family: raise a deficient-side connection group weaker than this x its mirror
MAX_FILL = 8.0
# Rule 2 (reference population gap): sensory groups short on BOTH sides of
# the male (or so short on one side that the mirror is no help), which only a
# reference animal reveals. Listed explicitly, with the evidence measured
# 2026-09-27 as neurons that connect (>= 20 output synapses), male vs FAFB,
# left / right:
#   JO-A/B (hearing)          75 / 9     vs 207 / 169
#   JO-C/E (wind, gravity)   203 / 119   vs 223 / 202
#   other JO                  65 / 34    vs 119 / 91
#   head bristles (BM)       333 / 238   vs 618 / 617
# ("JO-unclear" and "JO-A/B-unclear" male neurons have essentially no outputs:
# fragments.) Published: ~480 JO neurons per antenna (Kamikouchi et al. 2006
# J Comp Neurol 499:317). Male neurons carry more synapses each (a dataset
# difference), so the comparison is the group's TOTAL output per side against
# FAFB's times the overall male/FAFB synapse ratio S (median over well-connected
# type pairs); each side is raised to that (x at most MAX_REF_FILL), never lowered.
REFERENCE_GROUPS = {
    "JO-AB": ("JO-A", "JO-B"),
    "JO-CE": ("JO-C", "JO-E"),
    "JO-other": ("JO-",),
    "BM": ("BM",),
}
# (Not listed: photoreceptors R1-R8, 0.1-0.4 of FAFB, because the male dataset
# reconstructs only part of their axons and the model never drives them; ORNs
# are handled by rule 1.)
MAX_REF_FILL = 8.0
PROTECT = ("male-specific", "sexually dimorphic", "potentially sexually dimorphic", "potentially male-specific")


def _log(m):
    print("[merge] " + m, flush=True)


def _load(npz, index):
    z = np.load(npz, allow_pickle=False)
    shape = tuple(int(x) for x in z["shape"])
    w = sp.csr_matrix((z["data"].astype(np.int64), z["indices"], z["indptr"]), shape=shape)
    n = pd.read_csv(index, dtype={"root_id": np.int64})
    return n, w


def _sides(n, w):
    """Side per neuron; olfactory receptor neurons by the side of the PNs they
    mostly target (brain/sensory/orn_side.py), which also sides the unlabelled ones."""
    from brain.neurons.registry import Connectome
    from brain.sensory.orn_side import orn_sides
    c = Connectome(n, w, {"dataset": "tmp", "version": "0"})
    t = n["primary_type"].fillna("").astype(str).to_numpy()
    sd = n["side"].fillna("").astype(str).to_numpy()
    o = orn_sides(c)
    return np.where(np.char.startswith(t.astype(str), "ORN_") & (o != ""), o, sd)


def _edges(w, t, sd):
    pre = np.repeat(np.arange(w.shape[0]), np.diff(w.indptr))
    return pd.DataFrame({"pre": pre, "post": w.indices, "c": np.abs(w.data).astype(np.float64),
                         "tp": t[pre], "sp": sd[pre], "tq": t[w.indices], "sq": sd[w.indices]})


def _ncell(t, sd):
    return pd.Series(1, index=pd.MultiIndex.from_arrays([t, sd])).groupby(level=[0, 1]).sum()


def family(t: str) -> str:
    """ORN_VA1v -> ORN, JO-B1_a -> JO, BM_Vib -> BM, Mi1 -> Mi1 (the part
    before the FIRST separator of either kind)."""
    import re
    return re.split(r"[_-]", t, maxsplit=1)[0]


def _counts(t, sd):
    d = pd.DataFrame({"t": t, "s": sd})
    d = d[(d.t != "") & d.s.isin(["left", "right"])]
    c = d.groupby(["t", "s"]).size().unstack(fill_value=0)
    for s in ("left", "right"):
        if s not in c:
            c[s] = 0
    return c


def _l2(c):
    return np.log2((c["left"] + 1.0) / (c["right"] + 1.0))


def _synapse_scale(wm, tm, sdm, ftm, wf, tf, sdf) -> float:
    """Male / FAFB synapses per postsynaptic cell, median over FlyWire-matched
    type pairs with >= 5 per cell in both (1.24 on 2026-09-27)."""
    def pooled(w, t):
        pre = np.repeat(np.arange(w.shape[0]), np.diff(w.indptr))
        d = pd.DataFrame({"a": t[pre], "b": t[w.indices], "c": np.abs(w.data)})
        d = d[(d.a != "") & (d.b != "")]
        g = d.groupby(["a", "b"])["c"].sum()
        nc = pd.Series(1, index=t[t != ""]).groupby(level=0).sum()
        return g / nc.reindex(g.index.get_level_values(1)).to_numpy()
    pm, pf = pooled(wm, ftm), pooled(wf, tf)
    both = pm.index.intersection(pf.index)
    both = both[(pm[both] >= 5) & (pf[both] >= 5)]
    return float(np.median(pm[both] / pf[both]))


def population_gaps(tm, sdm, ftm, tf, sdf, protected) -> pd.DataFrame:
    """Male cell types (their own names) with a population gap (see module docstring)."""
    cm, cf = _counts(tm, sdm), _counts(tf, sdf)
    # the male type's FlyWire name, for the FAFB checks (the most common cross-reference)
    fmap = pd.Series(ftm, index=tm)[(tm != "") & (ftm != "")]
    fmap = fmap.groupby(level=0).agg(lambda s: s.value_counts().index[0])
    famm = _counts(np.array([family(x) if x else "" for x in tm]), sdm)
    ntypes = pd.Series([family(x) for x in set(tm) - {""}]).value_counts()
    famf = _counts(np.array([family(x) if x else "" for x in tf]), sdf)
    rows = []
    for ty, r in cm.iterrows():
        if ty in protected or ty not in fmap.index or fmap[ty] not in cf.index:
            continue
        rf = cf.loc[fmap[ty]]
        if rf["left"] + rf["right"] < MIN_CELLS:
            continue
        l2m, l2f = float(_l2(r)), float(_l2(rf))
        fam = family(ty)
        if fam not in famm.index or family(fmap[ty]) not in famf.index or ntypes.get(fam, 0) < FAMILY_MIN_TYPES:
            continue
        fl2m, fl2f = float(_l2(famm.loc[fam])), float(_l2(famf.loc[family(fmap[ty])]))
        if (abs(l2m) > COUNT_GAP_IN_FAMILY and abs(l2f) < FAFB_SYM and abs(fl2m) > FAMILY_GAP
                and np.sign(fl2m) == np.sign(l2m) and abs(fl2f) < FAFB_FAMILY_SYM and min(r["left"], r["right"]) > 0):
            weak = "left" if l2m < 0 else "right"
            strong = "right" if weak == "left" else "left"
            rows.append({"type": ty, "family": fam, "flywire_type": fmap[ty], "deficient_side": weak,
                         "n_deficient": int(r[weak]), "n_other": int(r[strong]),
                         "fafb_left": int(rf["left"]), "fafb_right": int(rf["right"]),
                         "factor": min(r[strong] / r[weak], MAX_FILL)})
    return pd.DataFrame(rows)


def symmetrize(W, t, sd, template=None):
    """Left/right balance (the robot must not drift or favour a side): for
    every connection group (pre type, side) -> (post type, side) the input per
    postsynaptic cell is set equal to its mirror image's, exact to within half
    a synapse per group (integer counts; _exact_totals).
      paired groups     both sides scaled to the mean of the two (gap-family
                        groups were already raised to their complete side)
      one-sided groups  halved, and a mirror copy added between the same types
                        on the other side (each connection's pre / post neuron
                        mapped to the neuron of the same rank in the mirror
                        type, its strength scaled for the cell counts)
      left as they are  types with neurons on one side only, and neurons with
                        no side (midline / unassigned)
    For presynaptic types in `template` ({type: side}; sensory populations
    with a demonstrated reconstruction gap) the better-reconstructed side is
    the template: its groups are kept, the deficient side's become their
    mirror images, template-only groups are copied across at full strength,
    and groups found only on the deficient side (not confirmed by the
    template) are dropped. (Review of the first version, 2026-09-27: "raise to
    the stronger side" copied an artefact -- the few left VA1v ORNs are mostly
    wired to the RIGHT PNs, x5.8 after the population fill -- onto both sides,
    and a backwards odour channel swamped odour steering.)
    Returns (W, stats)."""
    flip = {"left": "right", "right": "left"}
    E = _edges(W, t, sd)
    E["s"] = np.sign(W.data)
    ok = ((E.tp != "") & (E.tq != "") & E.sp.isin(["left", "right"]) & E.sq.isin(["left", "right"])).to_numpy()
    Eo = E[ok]
    nc = _ncell(t, sd)
    g = Eo.groupby(["tp", "sp", "tq", "sq"])["c"].sum()
    tp_, sp_, tq_, sq_ = (g.index.get_level_values(k) for k in range(4))
    msp, msq = np.array([flip[x] for x in sp_]), np.array([flip[x] for x in sq_])
    n_post = nc.reindex(pd.MultiIndex.from_arrays([tq_, sq_])).to_numpy()
    n_mpost = nc.reindex(pd.MultiIndex.from_arrays([tq_, msq])).fillna(0).to_numpy()
    n_mpre = nc.reindex(pd.MultiIndex.from_arrays([tp_, msp])).fillna(0).to_numpy()
    per = g.to_numpy() / n_post
    mper = pd.Series(per, index=g.index).reindex(pd.MultiIndex.from_arrays([tp_, msp, tq_, msq])).to_numpy()
    paired = ~np.isnan(mper)
    one = ~paired & (n_mpre > 0) & (n_mpost > 0)
    template = template or {}
    tmpl = np.array([template.get(x, "") for x in tp_], dtype=object)
    is_t = tmpl != ""
    on_def = is_t & (np.asarray(sp_) != tmpl)
    fac = np.ones(len(g))
    fac[paired & ~is_t] = 0.5 * (per[paired & ~is_t] + mper[paired & ~is_t]) / per[paired & ~is_t]
    fac[paired & on_def] = mper[paired & on_def] / per[paired & on_def]
    fac[one & ~is_t] = 0.5
    drop = ~paired & on_def                     # only on the deficient side: not confirmed
    fac[drop] = 0.0
    key = pd.MultiIndex.from_arrays([Eo.tp, Eo.sp, Eo.tq, Eo.sq])
    f_edge = np.ones(W.nnz)
    f_edge[Eo.index.to_numpy()] = pd.Series(fac, index=g.index).reindex(key).to_numpy()
    data = E["s"].to_numpy() * np.where(f_edge > 0, np.maximum(np.rint(E["c"].to_numpy() * f_edge), 1.0), 0.0)
    W1 = sp.csr_matrix((data, W.indices, W.indptr), shape=W.shape)
    # mirror copies of the one-sided groups
    gone = pd.Series(one & ~on_def, index=g.index)       # deficient-only groups are not copied
    Ea = Eo[gone.reindex(key).to_numpy()]
    order = np.lexsort((np.arange(len(t)), sd, t))
    rank = np.empty(len(t), np.int64)
    df = pd.DataFrame({"t": t[order], "s": sd[order]})
    rank[order] = df.groupby(["t", "s"]).cumcount().to_numpy()
    cells = {k: v.to_numpy() for k, v in pd.DataFrame({"t": t, "s": sd, "i": np.arange(len(t))}).groupby(["t", "s"])["i"]}
    mp = np.empty(len(Ea), np.int64)
    mq = np.empty(len(Ea), np.int64)
    ea_pre, ea_post = Ea.pre.to_numpy(), Ea.post.to_numpy()
    for (ty, side), idx in pd.Series(np.arange(len(Ea))).groupby([Ea.tp.to_numpy(), Ea.sp.to_numpy()]):
        c = cells[(ty, flip[side])]
        mp[idx.to_numpy()] = c[rank[ea_pre[idx.to_numpy()]] % len(c)]
    for (ty, side), idx in pd.Series(np.arange(len(Ea))).groupby([Ea.tq.to_numpy(), Ea.sq.to_numpy()]):
        c = cells[(ty, flip[side])]
        mq[idx.to_numpy()] = c[rank[ea_post[idx.to_numpy()]] % len(c)]
    npost_e = nc.reindex(pd.MultiIndex.from_arrays([Ea.tq, Ea.sq])).to_numpy()
    nmpost_e = nc.reindex(pd.MultiIndex.from_arrays([Ea.tq, [flip[x] for x in Ea.sq]])).to_numpy()
    share = np.where(np.array([x in template for x in Ea.tp.to_numpy()]), 1.0, 0.5)
    wa = np.maximum(np.rint(share * Ea.c.to_numpy() * nmpost_e / npost_e), 1.0) * Ea.s.to_numpy()
    A = sp.csr_matrix((wa, (mp, mq)), shape=W.shape)
    W2 = (W1 + A).tocsr()
    W2.sum_duplicates()
    W2.eliminate_zeros()
    W2, fixed = _exact_totals(W2, t, sd, template)
    unpaired = ~paired & ~one
    stats = {"groups_paired": int(paired.sum()), "groups_one_sided_mirrored": int(one.sum()),
             "groups_unmirrorable": int(unpaired.sum()),
             "gap_groups_dropped_deficient_only": int(drop.sum()),
             "gap_synapses_dropped": int(g.to_numpy()[drop].sum()),
             "synapses_in_unmirrorable_pct": round(100 * float(g.to_numpy()[unpaired].sum() / np.abs(W.data).sum()), 3),
             "synapses_without_side_pct": round(100 * float(E.c[~ok].sum() / E.c.sum()), 3),
             "mirror_connections_added": int(len(Ea)),
             "groups_total_corrected": fixed}
    return W2, stats


def _exact_totals(W, t, sd, template=None):
    """Synapse counts are integers, so scaling and rounding leave paired groups
    slightly off (review 2026-09-27: 6.8% of groups off by more than 7%).
    Set each paired group's total to its exact share of the mirrored mean by
    adjusting its largest connection (kept >= 1): per postsynaptic cell both
    sides then agree to within half a synapse per group."""
    flip = {"left": "right", "right": "left"}
    E = _edges(W, t, sd)
    ok = ((E.tp != "") & (E.tq != "") & E.sp.isin(["left", "right"]) & E.sq.isin(["left", "right"])).to_numpy()
    Eo = E[ok]
    nc = _ncell(t, sd)
    g = Eo.groupby(["tp", "sp", "tq", "sq"])["c"].sum()
    tp_, sp_, tq_, sq_ = (g.index.get_level_values(k) for k in range(4))
    n_post = nc.reindex(pd.MultiIndex.from_arrays([tq_, sq_])).to_numpy()
    per = g.to_numpy() / n_post
    mper = pd.Series(per, index=g.index).reindex(pd.MultiIndex.from_arrays(
        [tp_, [flip[x] for x in sp_], tq_, [flip[x] for x in sq_]])).to_numpy()
    paired = ~np.isnan(mper)
    template = template or {}
    tmpl = np.array([template.get(x, "") for x in tp_], dtype=object)
    is_t = tmpl != ""
    on_def = is_t & (np.asarray(sp_) != tmpl)
    target = np.where(~is_t, 0.5 * (per + mper), np.where(on_def, mper, per))
    delta = np.where(paired, np.rint(target * n_post - g.to_numpy()), 0.0)
    todo = pd.Series(delta, index=g.index)
    todo = todo[todo != 0]
    if not len(todo):
        return W, 0
    # the largest connection of each group to correct
    key = pd.MultiIndex.from_arrays([Eo.tp, Eo.sp, Eo.tq, Eo.sq])
    d_edge = todo.reindex(key).to_numpy()
    sel = ~np.isnan(d_edge)
    cand = pd.DataFrame({"pos": Eo.index.to_numpy()[sel], "c": Eo.c.to_numpy()[sel], "d": d_edge[sel],
                         "g": key[sel].to_flat_index()})
    big = cand.sort_values("c").groupby("g").tail(1)
    data = W.data.astype(np.float64)
    pos = big["pos"].to_numpy()
    mag = np.maximum(np.abs(data[pos]) + big["d"].to_numpy(), 1.0)
    data[pos] = np.sign(data[pos]) * mag
    W = sp.csr_matrix((data, W.indices, W.indptr), shape=W.shape)
    return W, int(len(big))


def build() -> dict:
    t0 = time.time()
    mdir = config.DERIVED_DIR / "malecns"
    nm, wm = _load(mdir / "connectome_malecns_v1.0.npz", mdir / "neuron_index_malecns_v1.0.csv.gz")
    nf, wf = _load(config.DERIVED_DIR / "connectome_v783.npz", config.DERIVED_DIR / "neuron_index_v783.csv.gz")
    _log("MaleCNS %d neurons %d connections; FAFB %d / %d" % (len(nm), wm.nnz, len(nf), wf.nnz))
    tm = nm["primary_type"].fillna("").astype(str).to_numpy()
    tf = nf["primary_type"].fillna("").astype(str).to_numpy()
    sdm, sdf = _sides(nm, wm), _sides(nf, wf)
    ftm = nm["flywire_type"].fillna("").astype(str).to_numpy()
    ftm = np.where(np.char.find(ftm.astype(str), ",") >= 0, "", ftm)       # ambiguous cross-references
    ftm = np.where(ftm == "", np.where(np.isin(tm, list(set(tf))), tm, ""), ftm)
    dim = pd.read_csv(mdir / "dimorphism_malecns_v1.0.csv.gz")
    prot_ids = set(dim.loc[dim["dimorphism"].isin(PROTECT), "root_id"])
    protected = set(tm[nm["root_id"].isin(prot_ids).to_numpy()]) - {""}

    gaps = population_gaps(tm, sdm, ftm, tf, sdf, protected)
    _log("population gaps: %d types %s" % (len(gaps), gaps["type"].tolist()))
    fac = np.ones(len(nm))
    for g in gaps.itertuples():
        fac[(tm == g.type) & (sdm == g.deficient_side)] = g.factor
    # rule 2: reference population gaps (see REFERENCE_GROUPS)
    def ref_group(t):
        g = np.full(len(t), "", object)
        for name, pre in REFERENCE_GROUPS.items():            # first match wins (JO-other last)
            g[(g == "") & np.array([x.startswith(pre) for x in t])] = name
        return g
    gm_, gf_ = ref_group(tm), ref_group(tf)
    out_m = np.asarray(abs(wm).sum(1)).ravel()
    out_f = np.asarray(abs(wf).sum(1)).ravel()
    S = _synapse_scale(wm, tm, sdm, ftm, wf, tf, sdf)
    ref = []
    for name in REFERENCE_GROUPS:
        for side in ("left", "right"):
            mm_, mf_ = (gm_ == name) & (sdm == side), (gf_ == name) & (sdf == side)
            tot_m, tot_f = float(out_m[mm_].sum()), float(out_f[mf_].sum()) * S
            if tot_m > 0 and tot_f > tot_m:
                f = min(tot_f / tot_m, MAX_REF_FILL)
                fac[mm_] *= f
                ref.append({"group": name, "side": side, "male_output_syn": int(tot_m),
                            "fafb_output_syn_scaled": int(tot_f), "factor": round(f, 3)})
    _log("reference population gaps: %s" % ref)
    pre = np.repeat(np.arange(wm.shape[0]), np.diff(wm.indptr))
    data = np.sign(wm.data) * np.rint(np.abs(wm.data) * fac[pre])
    W = sp.csr_matrix((data, wm.indices, wm.indptr), shape=wm.shape)
    # connection groups inside a gap family: the deficient side's remaining
    # neurons are also under-traced (fewer synapses each), so raise each of
    # their groups (type, side) -> (post type, post side) to its mirror image
    glog = []
    if len(gaps):
        fam_side = gaps.groupby("family")["deficient_side"].agg(lambda s: s.mode().iat[0]).to_dict()
        famv = np.array([family(x) if x else "" for x in tm])
        dsd = np.array([fam_side.get(f, "") for f in famv])
        in_fam = (dsd != "") & np.isin(sdm, ["left", "right"])
        E = _edges(W, tm, sdm)
        E = E[in_fam[E.pre] & np.isin(E.sq, ["left", "right"])]
        g = E.groupby(["tp", "sp", "tq", "sq"])["c"].sum()
        nc = _ncell(tm, sdm)
        per = pd.Series(g.to_numpy() / nc.reindex(pd.MultiIndex.from_arrays(
            [g.index.get_level_values(2), g.index.get_level_values(3)])).to_numpy(), index=g.index)
        flip = {"left": "right", "right": "left"}
        mper = per.reindex(pd.MultiIndex.from_arrays([g.index.get_level_values(0), [flip[x] for x in g.index.get_level_values(1)],
                                                      g.index.get_level_values(2), [flip[x] for x in g.index.get_level_values(3)]])).to_numpy()
        deficient = np.array([fam_side[family(tp)] == sp_ for tp, sp_, _, _ in g.index])
        raise_ = deficient & ~np.isnan(mper) & (per.to_numpy() * GROUP_GAP < mper)
        gf = pd.Series(np.where(raise_, np.minimum(mper / per.to_numpy(), MAX_FILL), 1.0), index=g.index)
        key = pd.MultiIndex.from_arrays([E.tp, E.sp, E.tq, E.sq])
        f_edge = np.ones(W.nnz)
        f_edge[E.index.to_numpy()] = gf.reindex(key).to_numpy()
        W = sp.csr_matrix((np.sign(W.data) * np.rint(np.abs(W.data) * f_edge), W.indices, W.indptr), shape=W.shape)
        glog = [(tp, sp_, tq, sq, round(a, 2), round(b, 2)) for (tp, sp_, tq, sq), a, b, r in
                zip(g.index, per.to_numpy(), mper, raise_) if r]
        _log("connection groups raised to their mirror inside gap families: %d" % len(glog))
    W.eliminate_zeros()
    # template side per gap type: rule 1 -> the family's complete side; rule 2
    # -> the side with more male output before filling
    template = {}
    famv = np.array([family(x) if x else "" for x in tm])
    if len(gaps):
        for fam, dside in gaps.groupby("family")["deficient_side"].agg(lambda s: s.mode().iat[0]).items():
            for ty in set(tm[famv == fam]) - {""}:
                template[ty] = "right" if dside == "left" else "left"
    for name in REFERENCE_GROUPS:
        tot = {sd_: float(out_m[(gm_ == name) & (sdm == sd_)].sum()) for sd_ in ("left", "right")}
        for ty in set(tm[gm_ == name]) - {""}:
            template.setdefault(ty, max(tot, key=tot.get))
    _log("template sides: %s" % {k: template[k] for k in sorted(template)[:6]})
    W, sym = symmetrize(W, tm, sdm, template)
    _log("symmetrized: %s" % sym)
    W.eliminate_zeros()
    W.sort_indices()

    out = config.DERIVED_DIR / "merged"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "connectome_merged_v1.npz", root_ids=nm["root_id"].to_numpy(), indptr=W.indptr,
                        indices=W.indices.astype(np.int32), data=W.data.astype(np.int32),
                        shape=np.array(W.shape), dataset=np.array(["merged MaleCNS v1.0 (gaps filled; ref. FAFB v783)"]))
    gaps.to_csv(out / "fill_log.csv.gz", index=False)
    pd.DataFrame(glog, columns=["pre_type", "pre_side", "post_type", "post_side", "before_per_cell", "mirror_per_cell"]
                 ).to_csv(out / "fill_log_groups.csv.gz", index=False)
    before, after = int(np.abs(wm.data).sum()), int(np.abs(W.data).sum())
    man = {"dataset": "merged MaleCNS (gaps filled)", "version": "1",
           "base": "Janelia MaleCNS v1.0 (brain/connectivity/build_malecns.py)",
           "reference": "FlyWire FAFB v783 (decides what is a gap; nothing copied from it)",
           "built_utc": datetime.now(timezone.utc).isoformat(),
           "rules": {"COUNT_GAP_IN_FAMILY": COUNT_GAP_IN_FAMILY, "FAMILY_GAP": FAMILY_GAP, "FAFB_SYM": FAFB_SYM,
                     "FAFB_FAMILY_SYM": FAFB_FAMILY_SYM, "MIN_CELLS": MIN_CELLS, "MAX_FILL": MAX_FILL,
                     "FAMILY_MIN_TYPES": FAMILY_MIN_TYPES, "GROUP_GAP": GROUP_GAP},
           "population_gaps": gaps.to_dict("records"),
           "reference_population_gaps": ref,
           "connection_groups_raised": len(glog),
           "symmetrization": sym,
           "evidence_2026_09_27": {
               "male_groups_lopsided_where_fafb_symmetric": 3640,
               "fafb_groups_lopsided_where_male_symmetric": 4135,
               "type_pairs_male_4x_below_female": 37864, "type_pairs_female_4x_below_male": 77044,
               "orn_by_wiring_male_left_right": [994, 1637], "orn_fafb_left_right": [1117, 1132]},
           "protected_types": len(protected),
           "n_neurons": int(len(nm)), "n_neuron_pairs": int(W.nnz), "n_synapses": after,
           "synapses_before": before, "synapses_changed_pct": round(100 * (after - before) / before, 3),
           "super_class_counts": {str(k): int(v) for k, v in nm.super_class.value_counts().items()}}
    (config.METADATA_DIR / "build_manifest_merged.json").write_text(json.dumps(man, indent=2, default=float))
    _log("synapses %d -> %d (+%.2f%%); %.0f s" % (before, after, man["synapses_changed_pct"], time.time() - t0))
    return man


if __name__ == "__main__":
    build()
