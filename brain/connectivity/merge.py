"""
The "complete map": the Janelia MaleCNS (brain + nerve cord of one male) with
its demonstrated reconstruction gaps filled. Selected with FLY_DATASET=merged.
FlyWire FAFB (the brain of one female) is the reference that decides what
counts as a gap and sets the fill targets; no FAFB connection is copied.
Three rules (details below and in results/complete_brain_2026-09-27/README.md):
  1. population gaps: under-reconstructed sensory types (the left antenna's
     ORNs) are mirrored from their complete side;
  2. senses short on both sides (hearing, wind, other Johnston's organ, head
     bristles): the better side, raised to FAFB's size (x the synapse scale),
     is mirrored as a whole group;
  3. left/right symmetrisation of every connection group (symmetrize), exact
     per postsynaptic cell; mirror copies that would be autapses are moved to
     the next cell of the type (move_autapses).
Dimorphic / male-specific types are never filled. Every change is logged
(fill_log.csv.gz for the population fills, fill_log_groups.csv.gz for groups;
counts in data/metadata/build_manifest_merged.json).

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
REFERENCE_GROUPS = ("sound", "wind", "JO-other", "BM")   # senses (brain/sensory/modality_map.json), rest of JO, head bristles
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


def sense_groups(t_male) -> np.ndarray:
    """Rule 2's groups for each MaleCNS neuron: "sound" / "wind" (the senses
    as mapped from FlyWire, modality_map.json), "JO-other", "BM", or ""."""
    import json as _json
    mmap = _json.loads((config.DERIVED_DIR / "malecns" / "modality_map.json").read_text())
    t = np.asarray(t_male).astype(str)
    g = np.full(len(t), "", object)
    g[np.asarray(mmap["sound"]["idx"])] = "sound"
    g[np.asarray(mmap["wind"]["idx"])] = "wind"
    g[(g == "") & np.char.startswith(t, "JO-")] = "JO-other"
    g[(g == "") & np.char.startswith(t, "BM")] = "BM"
    return g


def symmetry_types(t_male) -> np.ndarray:
    """The cell identity used for mirroring: the type, except that rule 2's
    sense groups are mirrored as whole groups."""
    g = sense_groups(t_male)
    out = np.asarray(t_male).astype(object).copy()
    out[g != ""] = np.array(["REF_" + x for x in g[g != ""]], dtype=object)
    return out.astype(str)


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


def symmetrize(W, t, sd, template=None, sign=None):
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
    # a mirror copy transmits with ITS presynaptic neuron's own sign; neurons
    # with no known transmitter (sign 0) send nothing (review 2026-09-27: the
    # source neuron's sign gave 16 sign-0 neurons 1,530 synapses)
    ms = np.sign(sign[mp]) if sign is not None else Ea.s.to_numpy()
    wa = np.maximum(np.rint(share * Ea.c.to_numpy() * nmpost_e / npost_e), 1.0) * ms
    keep_a = wa != 0
    A = sp.csr_matrix((wa[keep_a], (mp[keep_a], mq[keep_a])), shape=W.shape)
    W2 = (W1 + A).tocsr()
    W2.sum_duplicates()
    W2.eliminate_zeros()
    W2, fixed = _exact_totals(W2, t, sd, template)
    unpaired = ~paired & ~one
    stats = {"groups_paired": int(paired.sum()), "groups_one_sided_mirrored": int(one.sum()),
             "groups_unmirrorable": int(unpaired.sum()),
             "gap_groups_dropped_deficient_only": int(drop.sum()),
             "gap_synapses_dropped": int(g.to_numpy()[drop].sum()),
             "gap_synapses_dropped_by_group": {
                 str(k): int(v) for k, v in pd.Series(g.to_numpy()[drop], index=[
                     (x[4:] if x.startswith("REF_") else x.split("_")[0]) for x in np.asarray(tp_)[drop]
                 ]).groupby(level=0).sum().items()},
             "synapses_in_unmirrorable_pct": round(100 * float(g.to_numpy()[unpaired].sum() / np.abs(W.data).sum()), 3),
             "synapses_without_side_pct": round(100 * float(E.c[~ok].sum() / E.c.sum()), 3),
             "mirror_connections_added": int(keep_a.sum()),
             "mirror_connections_skipped_sign0": int((wa == 0).sum()),
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
    # the largest connection of each group to correct (the calibrated build;
    # spreading the correction proportionally instead changed the brain enough
    # to fail the exam's no_latching -- it would need re-calibration)
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


def move_autapses(W, W_raw, t, sd):
    """Mirror copies are mapped by rank, which wraps round when the mirror side
    has fewer cells of a type, so a connection between two cells of one type
    can land on one cell: an autapse the data does not have (review
    2026-09-27: 255 created; the raw MaleCNS has 26, which are kept).
      - the mirror side has 2+ cells of the type: the autapse moves to the next
        cell of the same type and side, so the group (type, side) -> (type,
        side) and the left/right balance are unchanged;
      - the mirror side has ONE cell: a within-type connection cannot exist
        there, so the group cannot be mirrored at all. It is treated like the
        other unmirrorable groups: no copy, and the source side's connections
        are restored to their raw strength (symmetrize had halved them for the
        copy; second review 2026-09-27).
    Done after the build rather than inside symmetrize() so that nothing else
    in the calibrated brain changes (re-routing inside it shifted
    _exact_totals' corrections and the exam). Returns (W, moved, restored
    groups)."""
    flip = {"left": "right", "right": "left"}
    W = W.tolil()
    d = W.diagonal()
    raw = W_raw.diagonal()
    cells = {k: v.to_numpy() for k, v in pd.DataFrame({"t": t, "s": sd, "i": np.arange(len(t))}).groupby(["t", "s"])["i"]}
    moved, single = 0, set()
    for i in np.flatnonzero((d != 0) & (raw == 0)):
        c = cells.get((t[i], sd[i]), np.array([i]))
        v = d[i]
        W[i, i] = 0
        if len(c) > 1:
            j = c[(int(np.searchsorted(c, i)) + 1) % len(c)]
            W[i, j] = W[i, j] + v
            moved += 1
        else:
            single.add((t[i], sd[i]))
    Wr = W_raw.tocsr()
    for ty, side in single:
        src = cells.get((ty, flip.get(side, "")), np.array([], np.int64))
        for a in src:
            row = Wr.getrow(a)
            for b, val in zip(row.indices, row.data):
                if b != a and b in set(src):
                    W[a, b] = val                    # the raw strength (and sign)
    W = W.tocsr()
    W.eliminate_zeros()
    return W, moved, len(single)


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
    # rule 2: reference population gaps, per SENSE (see REFERENCE_GROUPS):
    # hearing (sound), wind, other Johnston's organ, head bristles. The side
    # with more real output is the template; it is raised to the target --
    # the larger over the sides of max(its own output, FAFB's x S) -- and
    # symmetrize() then gives the other side its mirror image. No side is ever
    # lowered, and the mirror makes the two sides match per cell exactly.
    import json as _json
    mmap = _json.loads((mdir / "modality_map.json").read_text())
    gm_ = sense_groups(tm)
    gf_ = np.full(len(tf), "", object)
    gf_[np.isin(tf, mmap["sound"]["fafb_types"])] = "sound"
    gf_[np.isin(tf, mmap["wind"]["fafb_types"])] = "wind"
    gf_[(gf_ == "") & np.char.startswith(tf.astype(str), "JO-")] = "JO-other"
    gf_[(gf_ == "") & np.char.startswith(tf.astype(str), "BM")] = "BM"
    out_m = np.asarray(abs(wm).sum(1)).ravel()
    out_f = np.asarray(abs(wf).sum(1)).ravel()
    S = _synapse_scale(wm, tm, sdm, ftm, wf, tf, sdf)
    ref, template = [], {}
    for name in REFERENCE_GROUPS:
        raw = {sd_: float(out_m[(gm_ == name) & (sdm == sd_)].sum()) for sd_ in ("left", "right")}
        ref_f = {sd_: float(out_f[(gf_ == name) & (sdf == sd_)].sum()) * S for sd_ in ("left", "right")}
        target = max(max(raw[x], ref_f[x]) for x in ("left", "right"))
        tside = max(raw, key=raw.get)
        f = min(target / raw[tside], MAX_REF_FILL) if raw[tside] > 0 else 1.0
        fac[(gm_ == name) & (sdm == tside)] *= f
        template["REF_" + name] = tside
        ref.append({"group": name, "template_side": tside,
                    "male_output_syn": {k: int(v) for k, v in raw.items()},
                    "fafb_output_syn_scaled": {k: int(v) for k, v in ref_f.items()},
                    "target": int(target), "template_factor": round(f, 3)})
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
    # template side per gap type: rule 2's sense groups (set above) plus rule
    # 1's ORN types -> the family's complete side
    template = dict(template)
    famv = np.array([family(x) if x else "" for x in tm])
    if len(gaps):
        for fam, dside in gaps.groupby("family")["deficient_side"].agg(lambda s: s.mode().iat[0]).items():
            for ty in set(tm[famv == fam]) - {""}:
                template[ty] = "right" if dside == "left" else "left"
    # the sense groups are mirrored as WHOLE groups (their subtypes are typed
    # unevenly per side -- JO-A3 2/0, JO-mz 11/0 ...; review 2026-09-27)
    sign_m = nm["sign"].fillna(0).astype(int).to_numpy()
    W, sym = symmetrize(W, symmetry_types(tm), sdm, template, sign=sign_m)
    _log("symmetrized: %s" % sym)
    W, sym["autapses_moved"], sym["single_cell_groups_restored"] = move_autapses(W, wm, tm, sdm)
    _log("autapses created by the mirror copies: %d moved to the next cell of the type; %d "
         "within-type groups facing a single cell left unmirrored (raw strength)"
         % (sym["autapses_moved"], sym["single_cell_groups_restored"]))
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
