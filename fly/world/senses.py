"""
The fly's senses in the world: world state -> firing rates of REAL FlyWire
sensory neurons, every simulated millisecond.

Like brain/sensory/encoders.py, this is the only place where something other
than the connectome sets a firing rate. It never sets a rate for any neuron
that is not a sensory neuron, and it encodes nothing about what the fly
should do.

A. REAL DATA : which neurons exist and on which side (FlyWire v783); LC4/LPLC2
   receptive fields from the real column assignments (retinotopy.py).
B. PUBLISHED :
   - Olfaction: every ORN fires at its receptor's measured spontaneous rate,
     and odours change it by the measured responses (Hallem & Carlson 2006;
     brain/sensory/olfaction.py). Each fruit emits a natural blend
     (fermenting / ripe / mouldy / citrus), and a faint green-leaf odour is
     everywhere.
   - Sugar and bitter gustatory receptor neurons signal food quality on
     contact (brain/sensory/modalities.py).
   - Johnston's organ C and E neurons respond to sustained antennal
     deflection in opposite directions (Yorozu et al. 2009, Nature 458:201).
   - LC4 = angular velocity, LPLC2 = angular size of a looming object
     (encoders.LoomingEncoder).
C. OUR APPROXIMATIONS :
   - ORNs on each side sample the concentration at their own antenna (the
     connectome's ORNs are labelled by side of entry).
   - Wind: each antenna is deflected most by air from its own front quadrant
     (left antenna: front-left); deflection sign selects JO-C vs JO-E.
     Air speed includes the fly's own motion (headwind in flight). The
     response partly adapts to steady airflow: 30% tonic plus a transient
     component relative to the deflection over the last ~300 ms, so gusts
     and sudden airflow register strongly and a steady breeze weakly.
   - Taste: gustatory neurons fire while the fly stands on the fruit.
   - Taste reinforcement: tasting sugar activates reward dopamine neurons
     (PAM) and bitter activates punishment ones (PPL1) in real flies
     (Kirkhart & Scott 2015, J Neurosci 35:5950; sweet taste -> PAM-beta'2
     and PAM-gamma4: Yamagata et al. 2015, PNAS 112:578; Huetteroth et al.
     2015, Curr Biol 25:751). The model does not recover this multi-synaptic
     route (sugar changes DAN rates by < 1 Hz), so these dopamine neurons are
     driven directly while the fly tastes, at 100 Hz (as in optogenetic
     conditioning). FlyWire types by their MBON targets: PAM05/PAM06 (beta'2),
     PAM07/PAM08 (gamma4); PPL101 (gamma1pedc), PPL103 (gamma2alpha'1).
"""
from __future__ import annotations

import math

import numpy as np

from brain.sensory.encoders import LoomingEncoder
from brain.sensory.modalities import BY_KEY, resolve_neurons
from brain.sensory.olfaction import OlfactorySpace
from brain.sensory.retinotopy import load_retinotopy

TASTE_HZ = 120.0
REINFORCE_HZ = 100.0
REWARD_DANS = ("PAM05", "PAM06", "PAM07", "PAM08")
PUNISH_DANS = ("PPL101", "PPL103")
JO_MAX_HZ = 40.0            # static-deflection JO neurons: tens of Hz (C)
JO_HALF_AIRSPEED = 400.0     # mm/s
JO_TONIC = 0.3               # fraction of the response that does not adapt (C)
JO_ADAPT_MS = 300.0


class _PredatorView:
    """Adapter so LoomingEncoder can read the world's predator."""

    def __init__(self):
        self.st = {"active": False}

    def state(self, t_ms: float) -> dict:
        return self.st


class WorldSenses:
    """Sensory encoder for Session: rates_hz(t, world) for a fixed neuron set."""

    def __init__(self, connectome, world, body):
        self.c, self.world, self.body = connectome, world, body
        n = connectome.neurons
        t = n["primary_type"].fillna("").astype(str).to_numpy()
        side = n["side"].fillna("").astype(str).to_numpy()

        def by(mask):
            return np.flatnonzero(mask)

        groups = {}
        # olfactory receptor neurons, per glomerulus and side
        orn_types = sorted(set(x for x in t if x.startswith("ORN_")))
        self.olf = OlfactorySpace([x[4:] for x in orn_types])
        self._orn_groups = {"L": [], "R": []}
        for gi, ty in enumerate(orn_types):
            for sd, k in (("left", "L"), ("right", "R")):
                idx = np.flatnonzero((t == ty) & (side == sd))
                if len(idx):
                    name = "orn_%s_%s" % (ty[4:], k)
                    groups[name] = idx
                    self._orn_groups[k].append((name, gi))
        # blend of each fruit (odorant vector) and the leafy background
        from fly.world.world import FRUIT_KINDS
        self._fruit_blend = lambda f: self.olf.sources[FRUIT_KINDS[f.kind]["source"]]
        self._leaves = self.olf.sources["leaves"] * world.cfg.leaf_background
        groups["reward_dan"] = np.flatnonzero(np.isin(t, REWARD_DANS))
        groups["punish_dan"] = np.flatnonzero(np.isin(t, PUNISH_DANS))
        groups["sugar"] = np.asarray(resolve_neurons(BY_KEY["taste_sugar"], connectome), np.int64)
        groups["bitter"] = np.asarray(resolve_neurons(BY_KEY["taste_bitter"], connectome), np.int64)
        jo = np.asarray(resolve_neurons(BY_KEY["wind"], connectome), np.int64)
        is_c = np.char.startswith(t[jo].astype(str), "JO-C")
        is_e = np.char.startswith(t[jo].astype(str), "JO-E")
        for sd, k in (("left", "L"), ("right", "R")):
            groups["joC_" + k] = jo[is_c & (side[jo] == sd)]
            groups["joE_" + k] = jo[is_e & (side[jo] == sd)]
        self.loom = LoomingEncoder(connectome, load_retinotopy(connectome))
        self.pred_view = _PredatorView()
        groups["looming"] = self.loom.indices

        names = list(groups)
        idx = np.concatenate([groups[k] for k in names])
        gid = np.concatenate([np.full(len(groups[k]), i) for i, k in enumerate(names)])
        order = np.argsort(idx, kind="stable")
        if np.unique(idx).size != idx.size:
            raise ValueError("sensory groups overlap")
        self.indices = idx[order]
        self._ascending = np.arange(len(self.indices))        # already sorted
        self._names = names
        self._gid = gid[order]
        # position of the looming cells, in LoomingEncoder order
        self._loom_pos = np.searchsorted(self.indices, self.loom.indices)
        self.group_sizes = {k: int(len(v)) for k, v in groups.items()}
        npos = {k: i for i, k in enumerate(names)}
        self._orn_pos = {k: np.array([npos[nm] for nm, _ in v], int) for k, v in self._orn_groups.items()}
        self._orn_gi = {k: np.array([gi for _, gi in v], int) for k, v in self._orn_groups.items()}
        self._npos = npos
        self._B = None
        self._rates = np.zeros(len(self.indices))
        self.enabled = {"smell": True, "taste": True, "wind": True, "vision": True}
        self.reinforcement = True       # taste -> dopamine (see docstring)
        self.dt_ms = 1.0                # closed-loop interval (Session sets it)
        self._jo_adapt = np.zeros(4)    # adapted deflection, [C_L, E_L, C_R, E_R]

    # ------------------------------------------------------------------ rates
    def rates_hz(self, t_ms: float, world) -> np.ndarray:
        w, b = self.world, self.body
        s = b.state
        G = np.zeros(len(self._names))                 # rate per group
        senses = {}

        if self.enabled["smell"]:
            cl, cr = w.odour_at_antennae(b, self.dt_ms)   # per fruit
            if self._B is None or self._B.shape[0] != len(w.fruits):
                self._B = (np.array([self._fruit_blend(f) for f in w.fruits])
                           if w.fruits else np.zeros((0, len(self.olf.odorants))))
            for k, c in (("L", cl), ("R", cr)):
                r = self.olf.rates(self._leaves + c @ self._B)
                G[self._orn_pos[k]] = r[self._orn_gi[k]]
            senses["odour_L"], senses["odour_R"] = float(cl.sum()), float(cr.sum())
        else:
            for k in ("L", "R"):                          # smell "off": resting activity only
                G[self._orn_pos[k]] = self.olf.spont[self._orn_gi[k]]

        if self.enabled["taste"] and w._on_fruit is not None:
            if w._on_fruit.taste in ("sweet", "bitter"):
                sweet = w._on_fruit.taste == "sweet"
                G[self._npos["sugar" if sweet else "bitter"]] = TASTE_HZ
                if self.reinforcement:
                    G[self._npos["reward_dan" if sweet else "punish_dan"]] = REINFORCE_HZ
            senses["taste"] = w._on_fruit.taste

        if self.enabled["wind"]:
            # air velocity relative to the fly (world frame)
            hd = math.radians(s.heading_deg)
            wx, wy = w.wind_xy
            ax = wx - s.speed_mm_s * math.cos(hd)
            ay = wy - s.speed_mm_s * math.sin(hd)
            speed = math.hypot(ax, ay)
            if speed > 1.0:
                # direction the air comes FROM, head-centred (+ = right)
                from_world = math.degrees(math.atan2(-ay, -ax))
                phi = ((s.heading_deg - from_world) + 180.0) % 360.0 - 180.0
                mag = speed / (speed + JO_HALF_AIRSPEED)
                d = np.zeros(4)
                for q, pref in ((0, -45.0), (2, 45.0)):
                    defl = math.cos(math.radians(phi - pref))
                    d[q], d[q + 1] = mag * max(defl, 0.0), mag * max(-defl, 0.0)
                a = self._jo_adapt
                a += (d - a) * (self.dt_ms / JO_ADAPT_MS)
                # at an onset d - a = d, so the full response; it decays to the tonic part
                r = JO_MAX_HZ * (JO_TONIC * d + (1 - JO_TONIC) * np.maximum(d - a, 0.0))
                G[self._npos["joC_L"]], G[self._npos["joE_L"]] = r[0], r[1]
                G[self._npos["joC_R"]], G[self._npos["joE_R"]] = r[2], r[3]
                senses["air_from_deg"] = phi
                senses["airspeed_mm_s"] = speed

        rates = G[self._gid]
        if self.enabled["vision"]:
            self.pred_view.st = w.looming(b)
            if self.pred_view.st["active"]:
                rates[self._loom_pos] = self.loom.rates_hz(t_ms, self.pred_view)
                senses["loom_deg"] = self.pred_view.st["half_angle_deg"]
        w.senses = senses
        return rates

    def state(self, t_ms: float) -> dict:
        return {"kind": "world", "active": True, **self.world.senses_rounded()}

    @property
    def provenance(self) -> dict:
        return {"encoder": "fly/world/senses.py (closed-loop world)",
                "neurons": self.group_sizes,
                "olfaction": self.olf.provenance,
                "tuning": {"taste_hz": TASTE_HZ,
                           "jo_max_hz": JO_MAX_HZ, "jo_half_airspeed_mm_s": JO_HALF_AIRSPEED},
                "citations": ["Hallem & Carlson 2006, Cell 125:143",
                              "Semmelhack & Wang 2009, Nature 459:218",
                              "Yorozu et al. 2009, Nature 458:201",
                              "von Reyn et al. 2017, Nat Neurosci 20:1176"]}
