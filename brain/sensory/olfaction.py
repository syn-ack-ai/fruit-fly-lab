"""
Olfaction: odorant blends -> firing rates of the real olfactory receptor
neurons (ORNs) of each glomerulus, including their SPONTANEOUS activity.

PROVENANCE
----------
B. PUBLISHED DATA (data/external/Hallem_Carlson_2006.csv, from the drosolf
   package, github.com/tom-f-oconnell/drosolf):
   Hallem & Carlson 2006, Cell 125:143-160, doi:10.1016/j.cell.2006.01.050.
   For 24 odorant receptors (22 glomeruli): the spontaneous firing rate and
   the change in firing rate to 110 odorants at 10^-2 dilution, measured in
   the "empty neuron" system. Co-expressed Or33b and Or47a (both DM3) are
   summed; Or85b is VM5d.
   - Every ORN fires spontaneously (1-47 Hz); the rate is set by the receptor.
     Glomeruli not in the dataset get the dataset mean (13.8 Hz).
   - Apple cider vinegar also activates DM1 and VA2 (Or42b, Or92a; not in the
     dataset), which carry its attraction: Semmelhack & Wang 2009, Nature
     459:218. Added as a "vinegar_extra" component (response size C).
   - Geosmin (mould) activates only DA2 (Or56a): Stensmyr et al. 2012, Cell
     151:1345. Added as a "geosmin" component (response size C).
C. OUR APPROXIMATIONS
   - Concentration dependence: response = spontaneous + delta * h(c), with
     h(c) = c (1 + K) / (c + K), so c = 1 reproduces the published 10^-2
     response and responses saturate at high concentration. Rates are
     clipped at 0 (inhibitory odorants can silence an ORN).
   - Blends add linearly (real mixture interactions are not modelled).
   - The compositions of the natural sources below are approximate relative
     amounts of their main volatiles (fermenting fruit: e.g. Becher et al.
     2012, Funct Ecol 26:822; green leaves: C6 "green leaf volatiles";
     citrus peel: limonene, Dweck et al. 2013, Curr Biol 23:2472).
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[2] / "data" / "external" / "Hallem_Carlson_2006.csv"
HALF_SAT = 0.5                    # K in h(c)
DEFAULT_SPONT_HZ = 13.8           # dataset mean, for glomeruli it does not cover
EXTRA = {                         # components outside the dataset (C: response sizes)
    "vinegar_extra": {"DM1": 120.0, "VA2": 90.0},
    "geosmin": {"DA2": 150.0},
}

# Natural odour sources: odorant -> relative concentration at the source (C)
SOURCES = {
    "fermenting_fruit": {"ethanol": 1.0, "acetic acid": 0.6, "ethyl acetate": 0.5,
                         "isopentyl acetate": 0.25, "ethyl butyrate": 0.2,
                         "phenethyl alcohol": 0.2, "vinegar_extra": 1.0},
    "ripe_fruit": {"isopentyl acetate": 0.4, "ethyl butyrate": 0.3, "ethyl hexanoate": 0.2,
                   "hexyl acetate": 0.15, "ethanol": 0.2},
    "citrus": {"limonene": 1.0, "linalool": 0.2, "b-myrcene": 0.2},
    "leaves": {"E2-hexenal": 0.5, "Z3-hexenol": 0.5, "hexanal": 0.3, "E2-hexenyl acetate": 0.2},
    "mould": {"geosmin": 1.0, "1-octen-3-ol": 0.3},
}


class OlfactorySpace:
    def __init__(self, glomeruli: list[str], path: Path = DATA):
        rows = list(csv.reader(open(path)))
        gl = rows[0][1:25]
        rec = rows[1][1:25]
        gl = [g or {"33b": "DM3", "85b": "VM5d"}[r] for g, r in zip(gl, rec)]
        names = [r[0] for r in rows]
        spont_row = rows[names.index("spontaneous firing rate")][1:25]
        odorants = [r[0] for r in rows[2:] if r[0] and r[0] != "spontaneous firing rate"
                    and len(r) > 24 and r[1] not in ("",)]
        self.glomeruli = list(glomeruli)
        gpos = {g: i for i, g in enumerate(self.glomeruli)}
        self.odorants = odorants + list(EXTRA)
        opos = {o: i for i, o in enumerate(self.odorants)}
        self.spont = np.full(len(self.glomeruli), DEFAULT_SPONT_HZ)
        self.in_dataset = np.zeros(len(self.glomeruli), bool)
        self.delta = np.zeros((len(self.glomeruli), len(self.odorants)))
        seen = set()
        for col, g in enumerate(gl):
            if g not in gpos:
                continue
            i = gpos[g]
            s = float(spont_row[col])
            self.spont[i] = s if g not in seen else self.spont[i] + s
            self.in_dataset[i] = True
            for r in rows[2:]:
                if r[0] in opos and len(r) > col + 1 and r[col + 1] not in ("",):
                    self.delta[i, opos[r[0]]] += float(r[col + 1])
            seen.add(g)
        for comp, resp in EXTRA.items():
            for g, d in resp.items():
                if g in gpos:
                    self.delta[gpos[g], opos[comp]] = d
        self.sources = {k: self.blend(v) for k, v in SOURCES.items()}

    def blend(self, amounts: dict) -> np.ndarray:
        """Odorant -> relative concentration, as a vector over self.odorants."""
        v = np.zeros(len(self.odorants))
        idx = {o: i for i, o in enumerate(self.odorants)}
        for o, a in amounts.items():
            if o not in idx:
                raise KeyError("odorant %r not in Hallem & Carlson 2006 or EXTRA" % o)
            v[idx[o]] = a
        return v

    def rates(self, conc: np.ndarray) -> np.ndarray:
        """Per-glomerulus ORN firing rate (Hz) for odorant concentrations `conc`
        (c = 1 is the published 10^-2 dilution); includes spontaneous firing."""
        h = conc * (1.0 + HALF_SAT) / (conc + HALF_SAT)
        return np.maximum(0.0, self.spont + self.delta @ h)

    @property
    def provenance(self) -> dict:
        return {"data": "Hallem & Carlson 2006, Cell 125:143 (via drosolf)",
                "glomeruli_in_dataset": int(self.in_dataset.sum()),
                "glomeruli_total": len(self.glomeruli),
                "default_spontaneous_hz": DEFAULT_SPONT_HZ,
                "extra_components": EXTRA, "sources": SOURCES,
                "concentration_half_saturation": HALF_SAT}
