"""
Motor output: reading behavioural commands off real descending neurons (DNs).

IMPORTANT SCOPE LIMIT
---------------------
FlyWire FAFB v783 is a BRAIN connectome. The motor neurons that move legs and
wings are in the ventral nerve cord, which is NOT part of this dataset. The 110
neurons FlyWire labels `motor` innervate head structures (proboscis, antennae),
not the flight or leg muscles.

Therefore the last neural stage this project can simulate is the descending
neuron population (1,305 DNs in v783). That is the genuine output of the brain:
DNs are the only pathway from brain to ventral nerve cord. Everything past the
DNs -- muscles, legs, wings, body dynamics -- is a body model, and is labelled
as such in the UI.

PROVENANCE
----------
A. REAL DATA   : DN identities, cell types, sides, and all connectivity
                 driving them (FlyWire v783).
B. PUBLISHED   : the DN -> behaviour associations in DN_COMMANDS below. Each
                 entry carries its citation. These come from optogenetic
                 activation and silencing experiments, not from the connectome.
C. APPROX      : the mapping from DN firing rate to a 0-1 "command strength"
                 is a saturating function; thresholds are our choices.
D. ENGINEERING : the bookkeeping in this file.

Nothing here is a behavioural rule keyed to a stimulus. These functions never
see the stimulus; they see only DN spike counts produced by the simulation.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DNCommand:
    """A published association between a descending neuron type and behaviour."""
    cell_type: str
    channel: str          # motor channel this DN contributes to
    behaviour: str        # plain-language description
    laterality: str       # 'ipsilateral', 'bilateral', or 'none'
    citation: str
    doi: str


# --- B: published descending-neuron -> behaviour associations ---------------
# Only DN types with direct experimental evidence are listed. Any DN not in
# this table contributes to no behavioural channel and is reported by the UI as
# "Not currently modeled".
DN_COMMANDS = (
    DNCommand(
        "DNp01", "escape_takeoff",
        "Giant Fibre. Drives short-mode escape: mesothoracic leg extension and "
        "takeoff within ~5 ms, without preparatory wing raising.",
        "bilateral",
        "von Reyn et al. 2014, Nat Neurosci 17:962-965; "
        "Card & Dickinson 2008, J Exp Biol 211:341-353",
        "10.1038/nn.3741",
    ),
    DNCommand(
        "DNp02", "escape_long_mode",
        "Contributes to long-mode (GF-independent) escape with preparatory "
        "wing raising and a directed jump.",
        "ipsilateral",
        "Namiki et al. 2018, eLife 7:e34272; Cheong et al. 2024, eLife 13:RP96323",
        "10.7554/eLife.34272",
    ),
    DNCommand(
        "DNp04", "escape_long_mode",
        "Looming-responsive descending neuron in the escape network.",
        "ipsilateral",
        "Namiki et al. 2018, eLife 7:e34272",
        "10.7554/eLife.34272",
    ),
    DNCommand(
        "DNp11", "escape_long_mode",
        "Escape-network descending neuron projecting to leg neuropils.",
        "ipsilateral",
        "Namiki et al. 2018, eLife 7:e34272",
        "10.7554/eLife.34272",
    ),
    DNCommand(
        "DNp09", "forward_walk",
        "P9: drives forward walking with an ipsilateral turning component; "
        "used for object-directed steering. (Zacarias et al. 2018 also linked "
        "DNp09 to freezing; activation experiments show forward walking.)",
        "ipsilateral",
        "Bidaye et al. 2020, Neuron 108:469-485; Braun et al. 2024, Nature 630:686-694",
        "10.1016/j.neuron.2020.07.032",
    ),
    DNCommand(
        "DNg100", "forward_walk",
        "BDN2: command neuron for forward walking; activation initiates "
        "straight forward walking even in headless flies, and its activity "
        "sets stepping frequency (walking speed).",
        "bilateral",
        "Bidaye et al. 2020, Neuron 108:469-485; Pugliese et al. 2025/2026, "
        "bioRxiv 2025.09.12.675944 (connectome simulations: DNg100 drives the "
        "walking rhythm and controls stepping frequency)",
        "10.1016/j.neuron.2020.07.032",
    ),
    DNCommand(
        "DNp07", "landing",
        "Landing: activation drives leg extension for landing; its visual "
        "responses are gated on during flight.",
        "bilateral",
        "Ache et al. 2019, Nat Neurosci 22:1132-1139",
        "10.1038/s41593-019-0413-4",
    ),
    DNCommand(
        "DNp10", "landing",
        "Landing: activation drives landing leg extension (flight-gated).",
        "bilateral",
        "Ache et al. 2019, Nat Neurosci 22:1132-1139",
        "10.1038/s41593-019-0413-4",
    ),
    DNCommand(
        "DNg02_a", "flight_power",
        "DNg02 population: regulates wingbeat amplitude (flight power) by a "
        "population code; responds to visual motion in flight.",
        "ipsilateral",
        "Namiki et al. 2022, Curr Biol 32:1189-1196",
        "10.1016/j.cub.2022.01.008",
    ),
    DNCommand(
        "DNg02_b", "flight_power",
        "DNg02 population: regulates wingbeat amplitude (flight power) by a "
        "population code; responds to visual motion in flight.",
        "ipsilateral",
        "Namiki et al. 2022, Curr Biol 32:1189-1196",
        "10.1016/j.cub.2022.01.008",
    ),
    DNCommand(
        "DNg02_c", "flight_power",
        "DNg02 population: regulates wingbeat amplitude (flight power) by a "
        "population code; responds to visual motion in flight.",
        "ipsilateral",
        "Namiki et al. 2022, Curr Biol 32:1189-1196",
        "10.1016/j.cub.2022.01.008",
    ),
    DNCommand(
        "DNg02_d", "flight_power",
        "DNg02 population: regulates wingbeat amplitude (flight power) by a "
        "population code; responds to visual motion in flight.",
        "ipsilateral",
        "Namiki et al. 2022, Curr Biol 32:1189-1196",
        "10.1016/j.cub.2022.01.008",
    ),
    DNCommand(
        "DNg02_e", "flight_power",
        "DNg02 population: regulates wingbeat amplitude (flight power) by a "
        "population code; responds to visual motion in flight.",
        "ipsilateral",
        "Namiki et al. 2022, Curr Biol 32:1189-1196",
        "10.1016/j.cub.2022.01.008",
    ),
    DNCommand(
        "DNg02_f", "flight_power",
        "DNg02 population: regulates wingbeat amplitude (flight power) by a "
        "population code; responds to visual motion in flight.",
        "ipsilateral",
        "Namiki et al. 2022, Curr Biol 32:1189-1196",
        "10.1016/j.cub.2022.01.008",
    ),
    DNCommand(
        "DNg02_g", "flight_power",
        "DNg02 population: regulates wingbeat amplitude (flight power) by a "
        "population code; responds to visual motion in flight.",
        "ipsilateral",
        "Namiki et al. 2022, Curr Biol 32:1189-1196",
        "10.1016/j.cub.2022.01.008",
    ),
    DNCommand(
        "DNg02_h", "flight_power",
        "DNg02 population: regulates wingbeat amplitude (flight power) by a "
        "population code; responds to visual motion in flight.",
        "ipsilateral",
        "Namiki et al. 2022, Curr Biol 32:1189-1196",
        "10.1016/j.cub.2022.01.008",
    ),
    DNCommand(
        "DNge078", "groom",
        "aDN (FlyWire label 'putative aDN 2'): antennal grooming command; "
        "driven by antennal mechanosensation (Johnston's organ, head bristles).",
        "bilateral",
        "Hampel et al. 2015, eLife 4:e08758",
        "10.7554/eLife.08758",
    ),
    DNCommand(
        "DNa01", "turn",
        "Steering: activity biases the fly's turning.",
        "ipsilateral",
        "Rayshubskiy et al. 2024, Nature 631:135-143",
        "10.1038/s41586-024-07523-9",
    ),
    DNCommand(
        "DNa02", "turn",
        "Steering: unilateral activity drives ipsilateral turning.",
        "ipsilateral",
        "Rayshubskiy et al. 2024, Nature 631:135-143",
        "10.1038/s41586-024-07523-9",
    ),
    DNCommand(
        "MDN", "backward_walk",
        "Moonwalker descending neuron: drives backward walking.",
        "bilateral",
        "Bidaye et al. 2014, Science 344:97-101",
        "10.1126/science.1249964",
    ),
)

# Population readout (optional; FLY_POP_READOUT=1 or DescendingReadout(...,
# population=True)): commands the literature assigns to the same behaviour are
# pooled with the ones above, and channels read the pooled rate of all their
# cells across both hemispheres rather than the most active (type, side) group.
# Command DNs recruit networks of other DNs; complete behaviours need that
# co-activation (Braun et al. 2024, Nature 630:686).
POPULATION_EXTRA = (
    DNCommand(
        "DNg13", "turn",
        "Steering: lengthens strides on the outside of a turn; drives "
        "ipsiversive rotation like DNa02 (projects contralaterally).",
        "ipsilateral",
        "Yang et al. 2024, Cell 187:6290",
        "10.1016/j.cell.2024.08.033",
    ),
    DNCommand(
        "DNg97", "forward_walk",
        "oDN1: descending neuron downstream of P9 (DNp09) in the forward-"
        "walking pathway (FlyWire label 'P9-oDN1', Bidaye lab).",
        "ipsilateral",
        "Bidaye et al. 2020, Neuron 108:469 (P9 pathway)",
        "10.1016/j.neuron.2020.07.032",
    ),
    DNCommand(
        "DNg60", "halt",
        "Bluebell: SEZ stop neuron; halts walking.",
        "bilateral",
        "Sapkal et al. 2024, Nature (context-specific halting); Sterne et al. 2021",
        "10.1038/s41586-024-07854-7",
    ),
)
POOLED_CHANNELS = ("turn", "forward_walk", "backward_walk", "landing", "flight_power", "halt")

CHANNELS = tuple(sorted({d.channel for d in DN_COMMANDS + POPULATION_EXTRA}))
# cell types reported individually (lr_<type>, hz_<type>) by channels()
READOUT_TYPES = ("DNa01", "DNa02", "DNp09", "DNg100")

# C: rate at which a channel is considered fully driven (Hz, per DN).
CHANNEL_HALF_MAX_HZ = 60.0


def _mean(v: list) -> float:
    """float(np.mean(v)), without numpy's per-call overhead. For fewer than 8
    values numpy sums sequentially, so the result is bit-identical."""
    return sum(v) / len(v) if len(v) < 8 else float(np.mean(v))


class DescendingReadout:
    """Turns DN spiking in the simulation into motor channel activations."""

    def __init__(self, connectome, population: bool | None = None):
        self.c = connectome
        self.population = (os.environ.get("FLY_POP_READOUT", "0") == "1"
                           if population is None else bool(population))
        self.commands = []
        self.missing = []
        for cmd in DN_COMMANDS + (POPULATION_EXTRA if self.population else ()):
            cells = connectome.by_cell_type(cmd.cell_type)
            if cells.empty:
                self.missing.append(cmd.cell_type)
                continue
            self.commands.append((cmd, cells))

        # All descending neurons, for the "brain output" activity display.
        self.all_dn = connectome.neurons[
            connectome.neurons["super_class"].astype(str) == "descending"]
        self.dn_idx = self.all_dn["idx"].to_numpy(dtype=np.int64)

        # Index arrays per (cell_type, side)
        self.tracked = {}
        for cmd, cells in self.commands:
            for side in ("left", "right"):
                s = cells[cells["side"].astype(str) == side]
                self.tracked[(cmd.cell_type, side)] = s["idx"].to_numpy(dtype=np.int64)

        # Every tracked (type, side) group gathered into one index array, so a
        # frame's per-group spike sums are one gather + bincount rather than
        # ~40 small numpy calls (the readout runs every simulated millisecond).
        self._groups = [k for k, v in self.tracked.items() if v.size]
        self._gidx = np.concatenate([self.tracked[k] for k in self._groups])
        self._gid = np.repeat(np.arange(len(self._groups)),
                              [self.tracked[k].size for k in self._groups])
        self._gpos = {k: i for i, k in enumerate(self._groups)}
        # per channel: positions of its groups, split by side (for channels())
        gsize = np.array([self.tracked[k].size for k in self._groups], float)
        self._gscale = 1.0 / (gsize * 1e-3)            # spikes -> Hz per 1 ms of window
        chan_of = {c.cell_type: c.channel for c, _ in self.commands}
        self._chan_groups = {}
        for ch in CHANNELS:
            L = [i for i, (ct, sd) in enumerate(self._groups) if chan_of[ct] == ch and sd == "left"]
            R = [i for i, (ct, sd) in enumerate(self._groups) if chan_of[ct] == ch and sd == "right"]
            self._chan_groups[ch] = (L, R)
        self._type_groups = {}
        for i, (ct, sd) in enumerate(self._groups):
            if ct in READOUT_TYPES:
                LR = self._type_groups.setdefault(ct, ([], []))
                LR[0 if sd == "left" else 1].append(i)
        self._escape_groups = [(i, sd) for i, (ct, sd) in enumerate(self._groups)
                               if chan_of[ct].startswith("escape")]
        # public view of the grouping, for incremental window sums
        self.n_groups = len(self._groups)
        self.group_neurons = self._gidx
        self.group_ids = self._gid

        # Proboscis motor neurons are a genuine motor output that IS present in
        # the brain dataset (they innervate head muscles, not the VNC).
        # Labelled in FlyWire v783 by Claire McKellar.
        self.proboscis_idx = self._label_group_indices("proboscis_motor")

    def _label_group_indices(self, group: str) -> np.ndarray:
        try:
            from brain.neurons.labels import functional_group
            rids = functional_group(group)
        except Exception:
            return np.empty(0, dtype=np.int64)
        return np.array([self.c.idx(r) for r in rids
                         if int(r) in self.c._id2idx], dtype=np.int64)

    def proboscis_drive(self, spike_counts: np.ndarray, window_ms: float) -> float:
        """
        Saturating activation (0-1) of the real proboscis motor neurons.

        This is the one place the model reaches an actual motor neuron rather
        than stopping at a descending neuron.
        """
        if self.proboscis_idx.size == 0 or window_ms <= 0:
            return 0.0
        hz = float(spike_counts[self.proboscis_idx].sum()
                   / self.proboscis_idx.size / (window_ms * 1e-3))
        return hz / (hz + CHANNEL_HALF_MAX_HZ)

    def proboscis_drive_from_total(self, total_spikes: int, window_ms: float) -> float:
        """proboscis_drive() given the proboscis motor neurons' window total."""
        if self.proboscis_idx.size == 0 or window_ms <= 0:
            return 0.0
        hz = float(total_spikes / self.proboscis_idx.size / (window_ms * 1e-3))
        return hz / (hz + CHANNEL_HALF_MAX_HZ)

    # ------------------------------------------------------------------ read
    def group_sums(self, spike_counts: np.ndarray) -> np.ndarray:
        """Total spikes per tracked (type, side) group; exact integer sums."""
        return np.bincount(self._gid, weights=spike_counts[self._gidx],
                           minlength=len(self._groups))

    def rates(self, spike_counts: np.ndarray, window_ms: float) -> dict:
        """Firing rate (Hz) of every tracked DN type, per side."""
        if window_ms <= 0:
            window_ms = 1.0
        sums = self.group_sums(spike_counts)
        out = {}
        for (ctype, side), idx in self.tracked.items():
            if idx.size == 0:
                continue
            out["%s_%s" % (ctype, side)] = float(
                sums[self._gpos[(ctype, side)]] / idx.size / (window_ms * 1e-3))
        return out

    def channels(self, spike_counts: np.ndarray, window_ms: float,
                 sums: np.ndarray = None) -> dict:
        """
        Activation (0-1) of each published motor channel, plus turn bias.

        Channel activation is a saturating function of the mean firing rate of
        the DNs assigned to that channel. This is the only step that is not
        connectome-derived, and it introduces no stimulus dependence.
        """
        if window_ms <= 0:
            window_ms = 1.0
        if sums is None:
            sums = self.group_sums(spike_counts)
        hzv = np.asarray(sums, float) * self._gscale / window_ms
        act = (hzv / (hzv + CHANNEL_HALF_MAX_HZ)).tolist()
        hz = hzv.tolist()
        res = {}
        if self.population:
            return self._population_channels(sums, window_ms, act, hz)
        for ch, (L, R) in self._chan_groups.items():
            lv = [act[i] for i in L]
            rv = [act[i] for i in R]
            res[ch] = max(lv + rv) if (lv or rv) else 0.0
            if lv and rv:
                # right-minus-left activation (e.g. forward_walk_lr: DNp09
                # turns ipsilaterally; flight_power_lr: DNg02 asymmetry)
                res[ch + "_lr"] = sum(rv) / len(rv) - sum(lv) / len(lv)
        # per cell type: right-minus-left activation and mean rate (Hz), for
        # readouts that treat the types differently (e.g. DNa02 transient,
        # DNa01 sustained steering; Rayshubskiy et al. 2025, eLife 102230)
        for ct, (L_, R_) in self._type_groups.items():
            if L_ and R_:
                res["lr_" + ct] = (sum(act[i] for i in R_) / len(R_)
                                   - sum(act[i] for i in L_) / len(L_))
            allg = L_ + R_
            if allg:
                res["hz_" + ct] = float(sum(hz[i] for i in allg) / len(allg))
        # Ipsilateral steering convention: net bias toward the more active side.
        L, R = self._chan_groups.get("turn", ([], []))
        l = sum(act[i] for i in L) / len(L) if L else 0.0
        r = sum(act[i] for i in R) / len(R) if R else 0.0
        res["turn_bias"] = r - l          # >0 : turn right
        return res

    def _population_channels(self, sums, window_ms, act, hz) -> dict:
        """Pooled channels (see POPULATION_EXTRA): rate = all spikes of the
        channel's cells on a side / its cells / window; escape channels keep
        the single-cell (max) reading, as the Giant Fibre is one command cell."""
        sums = np.asarray(sums, float)
        size = 1.0 / (self._gscale * 1e-3)                  # cells per group
        f = lambda hz_: hz_ / (hz_ + CHANNEL_HALF_MAX_HZ)
        res = {}
        for ch, (L, R) in self._chan_groups.items():
            if not (L or R):
                res[ch] = 0.0
                continue
            if ch not in POOLED_CHANNELS:
                res[ch] = max(act[i] for i in L + R)
                if L and R:
                    res[ch + "_lr"] = (sum(act[i] for i in R) / len(R) - sum(act[i] for i in L) / len(L))
                continue
            pool = lambda g: float(sums[g].sum() / size[g].sum() / (window_ms * 1e-3)) if g else 0.0
            res[ch] = float(f(pool(L + R)))
            if L and R:
                res[ch + "_lr"] = float(f(pool(R)) - f(pool(L)))
        for ct, (L_, R_) in self._type_groups.items():
            if L_ and R_:
                res["lr_" + ct] = (sum(act[i] for i in R_) / len(R_) - sum(act[i] for i in L_) / len(L_))
            if L_ + R_:
                res["hz_" + ct] = float(sum(hz[i] for i in L_ + R_) / len(L_ + R_))
        res["turn_bias"] = res.get("turn_lr", 0.0)          # >0 : turn right
        return res

    def escape_laterality(self, spike_counts: np.ndarray,
                          sums: np.ndarray = None) -> float:
        """
        Left/right imbalance across escape DNs, in [-1, 1] (>0 = right side more
        active). Used by the body model to direct the escape jump.
        """
        if sums is None:
            sums = self.group_sums(spike_counts)
        tot = {"left": 0.0, "right": 0.0}
        for i, sd in self._escape_groups:
            tot[sd] += float(sums[i]) * self._gscale[i] * 1e-3
        s = tot["left"] + tot["right"]
        return 0.0 if s == 0 else (tot["right"] - tot["left"]) / s

    # ----------------------------------------------------------- provenance
    @property
    def provenance(self) -> dict:
        return {
            "n_descending_neurons_in_dataset": int(len(self.all_dn)),
            "n_dn_types_in_dataset": int(self.all_dn["primary_type"].nunique()),
            "modelled_dn_types": [c.cell_type for c, _ in self.commands],
            "not_modelled_note": (
                "The other DN types in FlyWire v783 have no established "
                "behavioural assignment used here and are reported as "
                "'Not currently modeled'."),
            "vnc_limitation": (
                "Leg and wing motor neurons are in the ventral nerve cord and "
                "are absent from FlyWire FAFB v783. Simulation ends at the "
                "descending neurons."),
            "commands": [
                {"cell_type": c.cell_type, "channel": c.channel,
                 "behaviour": c.behaviour, "citation": c.citation, "doi": c.doi}
                for c, _ in self.commands
            ],
        }
