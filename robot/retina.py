"""
The fly's own eyes as the input (option A of experiments/vision_ab): what the
robot sees drives the connectome's PHOTORECEPTORS, and the optic lobes (and
everything after them) compute from there, through the real wiring.

    scene (a panorama: azimuth -180..180, elevation -90..90, brightness 0..1)
      -> each photoreceptor samples it in its viewing direction (an
         acceptance angle of ~5 degrees)
      -> its rate: a resting rate plus the brightness contrast against its own
         slowly adapting mean (light up: more transmitter)
      -> R1-R6 (lamina), R7 / R8 (medulla) -> ... -> LC / LPLC / LPTC -> central brain

PROVENANCE
A. REAL DATA: the photoreceptor neurons (merged brain: R1-R6, R7*, R8*; their
   transmitter, histamine, is inhibitory onto L1 / L2 as in the fly) and the
   optic-lobe column map (brain/sensory/retinotopy.py). Photoreceptors have
   no column assignment of their own: a photoreceptor's viewing direction is
   the synapse-weighted mean direction of the column-assigned neurons it
   drives (R1-R6 -> L1-L3; R7 / R8 -> Mi1, Tm9, Tm20, ...).
B. PUBLISHED: photoreceptors respond to light with graded depolarisation and
   adapt to the mean (e.g. Juusola & Hardie 2001); acceptance angle ~5 deg
   (Stavenga 2003); the box-filter / hexagonal sampling of images onto the eye
   follows flyvis (Lappalainen et al. 2024, Nature 634:1132).
C. APPROXIMATIONS: graded photoreceptor signals are drawn as Poisson spike
   rates (the whole-brain model's sensory convention); one gain and one
   adaptation time for all photoreceptor types (no colour or polarisation:
   R7 / R8 see the same brightness); the lamina's own amacrine circuitry is
   only what the connectome has. Parameters below are first guesses, to be
   tuned against the optic lobe's known responses (experiments/vision_ab).
"""
from __future__ import annotations

import numpy as np

PHOTORECEPTOR_TYPES = ("R1-R6", "R7y", "R7p", "R7d", "R7_unclear", "R8y", "R8p", "R8d", "R8_unclear",
                       "R7R8_unclear")
PANO_W, PANO_H = 180, 90            # 2 degrees a pixel: azimuth -180..180, elevation 90..-90
ACCEPTANCE_DEG = 5.0                # photoreceptor acceptance angle (FWHM)
REST_HZ = 10.0                      # a photoreceptor in steady light
GAIN_HZ = 120.0                     # per unit of brightness contrast
MAX_HZ = 250.0
ADAPT_S = 0.3                       # the adapting mean's time constant


def pano_index(az_deg, el_deg):
    """Fractional pixel (x, y) of directions in the panorama."""
    x = (np.asarray(az_deg) + 180.0) / 360.0 * PANO_W - 0.5
    y = (90.0 - np.asarray(el_deg)) / 180.0 * PANO_H - 0.5
    return x, y


def blur(img, sigma_px):
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(img, sigma_px, mode=("nearest", "wrap"))


def photoreceptor_directions(connectome, retinotopy) -> dict:
    """{neuron index: (azimuth, elevation)} for every photoreceptor whose
    column-assigned targets give it a direction."""
    c, cols = connectome, retinotopy.columns
    rid2idx = dict(zip(c.neurons["root_id"], c.neurons["idx"]))
    col_idx = cols["root_id"].map(rid2idx)
    ok = col_idx.notna()
    known = np.zeros(c.n, bool)
    az = np.zeros(c.n)
    el = np.zeros(c.n)
    ii = col_idx[ok].astype(int).to_numpy()
    known[ii], az[ii], el[ii] = True, cols.loc[ok, "azimuth_deg"].to_numpy(), cols.loc[ok, "elevation_deg"].to_numpy()
    t = c.neurons["primary_type"].fillna("").astype(str).to_numpy()
    w = c.w.tocsr()
    out = {}
    for i in np.flatnonzero(np.isin(t, PHOTORECEPTOR_TYPES)):
        tgt = w.indices[w.indptr[i]:w.indptr[i + 1]]
        syn = np.abs(w.data[w.indptr[i]:w.indptr[i + 1]]).astype(float)
        m = known[tgt]
        if syn[m].sum() <= 0:
            continue
        wt = syn[m] / syn[m].sum()
        out[int(i)] = (float(wt @ az[tgt[m]]), float(wt @ el[tgt[m]]))
    return out


class RetinaEncoder:
    """Session encoder: a panorama source -> photoreceptor rates (see module
    docstring). `source(t_ms)` returns (frame_id, panorama (PANO_H, PANO_W)
    float32 0..1); the rates change only when the frame does."""

    CELL_TYPES = PHOTORECEPTOR_TYPES

    def __init__(self, connectome, retinotopy, source, rest_hz=REST_HZ, gain_hz=GAIN_HZ, adapt_s=ADAPT_S):
        d = photoreceptor_directions(connectome, retinotopy)
        idx = np.array(sorted(d), np.int64)
        self.indices = idx
        self.az = np.array([d[i][0] for i in idx])
        self.el = np.array([d[i][1] for i in idx])
        self._x, self._y = pano_index(self.az, self.el)
        self.source = source
        self.rest, self.gain, self.adapt_s = float(rest_hz), float(gain_hz), float(adapt_s)
        self.sigma_px = ACCEPTANCE_DEG / 2.355 / (360.0 / PANO_W)
        self._mean = None
        self._last = (None, None, None)              # (frame id, t_ms, rates)
        self.last = {}

    def sample(self, img) -> np.ndarray:
        """Each photoreceptor's brightness: the blurred panorama in its direction."""
        from scipy.ndimage import map_coordinates
        b = blur(np.asarray(img, np.float32), self.sigma_px)
        return map_coordinates(b, [self._y, self._x], order=1, mode="nearest")

    def rates_hz(self, t_ms: float, stim=None) -> np.ndarray:
        fid, img = self.source(t_ms)
        if fid == self._last[0]:
            return self._last[2]
        lum = self.sample(img)
        if self._mean is None:
            self._mean = lum.copy()
        dt = 0.0 if self._last[1] is None else max(0.0, (t_ms - self._last[1]) / 1000.0)
        contrast = (lum - self._mean) / (self._mean + 0.05)
        rates = np.clip(self.rest + self.gain * contrast, 0.0, MAX_HZ)
        self._mean += (lum - self._mean) * (1.0 - np.exp(-dt / self.adapt_s))
        rates.flags.writeable = False                   # Session._rates_at: unchanged by identity
        self._last = (fid, t_ms, rates)
        self.last = {"mean_hz": round(float(rates.mean()), 1), "max_hz": round(float(rates.max()), 1)}
        return rates

    def state(self, t_ms: float) -> dict:
        return {"kind": "retina", "active": True, **self.last}

    @property
    def provenance(self) -> dict:
        return {"drives": {"photoreceptors (R1-R8)": int(len(self.indices))},
                "source": "connectome photoreceptors; directions from column-assigned targets (robot/retina.py)"}


class LaminaEncoder(RetinaEncoder):
    """Option A2 / A3: the lamina's monopolar cells L1-L3 driven with the
    sign of their real graded responses (they hyperpolarise to light and
    depolarise to dark: Laughlin & Hardie 1978; Clark et al. 2011), each in
    its own column's direction (MaleCNS columns; no photoreceptor step).

    The whole-brain model's neurons rest silent below threshold, so release
    from inhibition (the ON pathway: L1 -> Mi1 / Tm3 through glutamate) cannot
    show; `tonic_hz` adds a resting drive to the medulla's columnar neurons
    (A3) so that it can. C: one rate rule for all three types."""

    LAMINA = ("L1", "L2", "L3")
    TONIC = ("Mi1", "Mi4", "Mi9", "Tm1", "Tm2", "Tm4", "Tm9", "Tm20", "C2", "C3", "T1", "L5")

    def __init__(self, connectome, retinotopy, source, rest_hz=30.0, gain_hz=60.0, adapt_s=ADAPT_S,
                 tonic_hz=0.0):
        cols = retinotopy.columns
        rid2idx = dict(zip(connectome.neurons["root_id"], connectome.neurons["idx"]))
        cols = cols.assign(idx=cols["root_id"].map(rid2idx)).dropna(subset=["idx"])
        lam = cols[cols["type"].isin(self.LAMINA)]
        self.n_lamina = len(lam)
        idx = lam["idx"].astype(int).to_numpy()
        az, el = lam["azimuth_deg"].to_numpy(), lam["elevation_deg"].to_numpy()
        self.tonic_hz = float(tonic_hz)
        if self.tonic_hz > 0:
            ton = cols[cols["type"].isin(self.TONIC)]
            idx = np.concatenate([idx, ton["idx"].astype(int).to_numpy()])
            az = np.concatenate([az, ton["azimuth_deg"].to_numpy()])
            el = np.concatenate([el, ton["elevation_deg"].to_numpy()])
        order = np.argsort(idx)
        self.indices, self.az, self.el = idx[order], az[order], el[order]
        self._is_lamina = order < self.n_lamina
        self._x, self._y = pano_index(self.az, self.el)
        self.source = source
        self.rest, self.gain, self.adapt_s = float(rest_hz), float(gain_hz), float(adapt_s)
        self.sigma_px = ACCEPTANCE_DEG / 2.355 / (360.0 / PANO_W)
        self._mean = None
        self._last = (None, None, None)
        self.last = {}

    def rates_hz(self, t_ms: float, stim=None) -> np.ndarray:
        fid, img = self.source(t_ms)
        if fid == self._last[0]:
            return self._last[2]
        lum = self.sample(img)
        if self._mean is None:
            self._mean = lum.copy()
        dt = 0.0 if self._last[1] is None else max(0.0, (t_ms - self._last[1]) / 1000.0)
        contrast = (lum - self._mean) / (self._mean + 0.05)
        rates = np.where(self._is_lamina, np.clip(self.rest - self.gain * contrast, 0.0, MAX_HZ), self.tonic_hz)
        self._mean += (lum - self._mean) * (1.0 - np.exp(-dt / self.adapt_s))
        rates.flags.writeable = False
        self._last = (fid, t_ms, rates)
        self.last = {"lamina_mean_hz": round(float(rates[self._is_lamina].mean()), 1)}
        return rates

    @property
    def provenance(self) -> dict:
        return {"drives": {"L1-L3 (lamina)": self.n_lamina,
                           "medulla columnar (tonic)": int((~self._is_lamina).sum())},
                "source": "MaleCNS optic-lobe columns; graded sign of L1-L3 (robot/retina.py)"}
