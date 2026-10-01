"""
The fly's eyes through a published model of its visual system (option B of
experiments/vision_ab): flyvis (Lappalainen et al. 2024, Nature 634:1132;
TuragaLab/flyvis, MIT licence) computes the optic lobe from the visual field,
and its activity drives the matching neurons of the whole-brain connectome,
which carries it on (lobula, lobula plate, central brain).

    scene (a panorama: azimuth -180..180, elevation 90..-90, brightness 0..1)
      -> each eye: 721 hexagonal columns, 5.8 deg apart (flyvis's lattice),
         around the eye's centre (-55 / +55 deg, as brain/sensory/retinotopy.py)
      -> flyvis: 65 cell types x 721 columns, photoreceptors to T4/T5, Tm, TmY
         (a connectome-constrained network trained on optic flow; both eyes as
         a batch of two; Euler steps of 10 ms, on the GPU when there is one)
      -> each connectome neuron of a flyvis type takes the activity of its
         type at the flyvis column nearest its own viewing direction
      -> Poisson rates for the connectome model

PROVENANCE
A. REAL DATA: the connectome's optic-lobe neurons and their column map
   (retinotopy.py; types without a column: the synapse-weighted mean
   direction of their column-assigned inputs).
B. PUBLISHED: flyvis's pretrained network (ensemble flow/0000, model 000);
   its lattice orientation, measured here on gratings: T4a / T5a prefer
   motion towards -x of flyvis's hex image (so -x is posterior), T4c / T5c
   towards -y (so -y is up), as in the fly (Maisak et al. 2013).
C. APPROXIMATIONS: flyvis's activities are graded (voltage-like, arbitrary
   units); a neuron's rate is RATE_MAX x (activity above its grey-screen
   level) / (its type's 99th-percentile response to calibration gratings and
   flashes), clipped to 0..1.5 x RATE_MAX. Connectome neurons further than
   1.5 columns from any flyvis column (the eye's edge) get no drive.
   Photoreceptors are not driven (flyvis does that stage); one flyvis model,
   not the ensemble mean. Brightness goes to all eight photoreceptor inputs,
   after light adaptation (photoreceptors adapt to the mean: Juusola & Hardie
   2001): each eye's input is scaled so its slowly tracked mean brightness
   (ADAPT_S) sits at flyvis's grey (0.5), as the rates are measured from the
   grey-screen state (without it a bright room drives the ON pathway and HS
   ~100 Hz all the time).
"""
from __future__ import annotations

import os
import time
import warnings

import numpy as np

from robot.retina import PANO_W, pano_index

os.environ.setdefault("FLYVIS_ROOT_DIR", os.path.expanduser("~/milo/models/flyvis"))
MODEL = "flow/0000/000"
DT = float(os.environ.get("FLY_EYE_DT", 0.01))   # s, one flyvis step (its recommendation: <= 1/50)
OMM_DEG = 5.8                   # flyvis's interommatidial angle
EYE_CENTRE = {"left": -55.0, "right": +55.0}
RATE_MAX = float(os.environ.get("FLY_EYE_RATE_MAX", 100.0))   # Hz at a type's 99th-percentile response
EDGE_COLUMNS = 1.5              # beyond this many columns from the lattice: no drive
ADAPT_S = 0.5                   # light adaptation: each eye's mean brightness is brought to grey
BASELINE_S = float(os.environ.get("FLY_EYE_BASELINE_S", 0) or 0) or None
                                # a neuron's rate from its activity above its own adapting
                                # baseline (s), instead of its grey-screen level (robot/eye.py: 2 s)
ALIASES = {"CT1(Lo1)": ["CT1"], "Am": ["Am1"], "TmY9": ["TmY9a", "TmY9b"]}
SKIP = {"R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "CT1(M10)"}


def hexal_directions(u, v, eye):
    """flyvis hex (u, v) -> (azimuth, elevation) for the left / right eye.
    flyvis's hex image: x = v, y = u + v/2 (columns of 5.8 deg; BoxEye's
    layout); +x is anterior, +y is down (measured, see the module docstring)."""
    x, y = np.asarray(v, float) * OMM_DEG, (np.asarray(u, float) + np.asarray(v, float) / 2.0) * OMM_DEG
    if eye == "right":
        az = EYE_CENTRE["right"] - x           # anterior = towards 0 deg
    else:
        az = EYE_CENTRE["left"] + x
    return az, -y


def neuron_directions(connectome, retinotopy, types) -> dict:
    """{neuron index: (azimuth, elevation)} for neurons of `types`: from the
    column map, else the synapse-weighted mean direction of their
    column-assigned inputs."""
    c, cols = connectome, retinotopy.columns
    rid2idx = dict(zip(c.neurons["root_id"], c.neurons["idx"]))
    ci = cols["root_id"].map(rid2idx)
    ok = ci.notna()
    known = np.zeros(c.n, bool)
    az = np.zeros(c.n)
    el = np.zeros(c.n)
    ii = ci[ok].astype(int).to_numpy()
    known[ii], az[ii], el[ii] = True, cols.loc[ok, "azimuth_deg"].to_numpy(), cols.loc[ok, "elevation_deg"].to_numpy()
    t = c.neurons["primary_type"].fillna("").astype(str).to_numpy()
    wt = c.w.T.tocsr()                          # rows: post; columns: pre
    out = {}
    for i in np.flatnonzero(np.isin(t, list(types))):
        if known[i]:
            out[int(i)] = (float(az[i]), float(el[i]))
            continue
        pre = wt.indices[wt.indptr[i]:wt.indptr[i + 1]]
        syn = np.abs(wt.data[wt.indptr[i]:wt.indptr[i + 1]]).astype(float)
        m = known[pre]
        if syn[m].sum() <= 0:
            continue
        w = syn[m] / syn[m].sum()
        out[int(i)] = (float(w @ az[pre[m]]), float(w @ el[pre[m]]))
    return out


def sampling_matrix(x, y, sigma_px, shape=None):
    """Sparse (len(x), H * W) weights: a normalised Gaussian (sigma in pixels,
    to 3 sigma) around each fractional pixel (x, y) of the panorama; columns
    wrap, rows are clipped to the image."""
    from scipy.sparse import csr_matrix
    from robot.retina import PANO_H
    H, W = shape or (PANO_H, PANO_W)
    r = int(np.ceil(3 * sigma_px))
    dx, dy = np.meshgrid(np.arange(-r, r + 2), np.arange(-r, r + 2))
    px = np.floor(x)[:, None] + dx.ravel()[None, :]
    py = np.floor(y)[:, None] + dy.ravel()[None, :]
    w = np.exp(-((px - x[:, None]) ** 2 + (py - y[:, None]) ** 2) / (2 * sigma_px ** 2))
    w[(py < 0) | (py >= H)] = 0.0
    w /= w.sum(1, keepdims=True)
    cols = (np.clip(py, 0, H - 1) * W + np.mod(px, W)).astype(np.int64)
    rows = np.repeat(np.arange(len(x)), w.shape[1])
    return csr_matrix((w.ravel().astype(np.float32), (rows, cols.ravel())), shape=(len(x), H * W))


class FlyvisEncoder:
    """Session encoder (robot/retina.RetinaEncoder's interface): a panorama
    source -> flyvis -> rates of the connectome's optic-lobe neurons.
    `source(t_ms)` returns (frame_id, panorama). flyvis steps every DT of
    simulated time (with the latest frame); the rates change only when it
    has stepped (the same read-only array otherwise)."""

    def __init__(self, connectome, retinotopy, source, model: str = MODEL, calibrate: bool = True,
                 saved: str | None = None, baseline_s: float | None = BASELINE_S):
        """`saved`: a mapping written by save() (robot/eye.py builds it once),
        instead of the connectome and retinotopy (then both may be None)."""
        import torch
        import flyvis
        from flyvis import NetworkView
        self.torch = torch
        self.source = source
        self.model = model
        self.net = NetworkView(flyvis.results_dir / model).init_network(checkpoint="best")
        self.net.eval()
        fc = self.net.connectome
        self.ntype = np.array(fc.nodes.type[:]).astype(str)
        u, v = np.array(fc.nodes.u[:]), np.array(fc.nodes.v[:])
        self.inp = np.asarray(self.net.stimulus.input_index)           # (8, 721) node indices
        hu, hv = u[self.inp[0]], v[self.inp[0]]
        self.eye_dir = {e: hexal_directions(hu, hv, e) for e in ("left", "right")}
        self._px = {e: pano_index(*self.eye_dir[e]) for e in self.eye_dir}
        self.sigma_px = OMM_DEG / 2.355 / (360.0 / PANO_W)
        if saved is not None:
            m = np.load(saved, allow_pickle=False)
            if str(m["model"]) != model or len(m["scale"]) != len(self.ntype):
                raise ValueError(f"{saved}: made for flyvis model {m['model']}, not {model}")
            self.indices, self._eye, self._node = m["indices"], m["eye"], m["node"]
            self._ty = self.ntype[self._node]
        else:
            self._map(connectome, retinotopy, u, v)
        self.n_by_type = {ft: int(n) for ft, n in zip(*np.unique(self._ty, return_counts=True))}
        with torch.no_grad():
            self._params = self.net._param_api()
            self._state = self.net.steady_state(1.0, DT, batch_size=2, value=0.5)
        self._a0 = self._state.nodes.activity.detach().cpu().numpy()            # grey-screen activity (2, nodes)
        self._scale = np.ones(len(self.ntype))
        if saved is not None:
            self._scale = np.asarray(m["scale"], float)
        elif calibrate:
            self._calibrate()
        self.baseline_s = baseline_s
        self._base = None                                                       # adapting baseline per mapped neuron
        self._t_ms = None
        self._eye_mean = None                                                   # adapted brightness per eye
        self._rates = self._to_rates(self._a0)
        self.step_ms_total, self.steps = 0.0, 0
        self.last = {}

    def _map(self, connectome, retinotopy, u, v):
        """Connectome neurons per flyvis type, each at its nearest flyvis
        column of its own eye."""
        ftypes = [x for x in np.unique(self.ntype) if x not in SKIP]
        names = {ft: ALIASES.get(ft, [ft]) for ft in ftypes}
        cname = connectome.neurons["primary_type"].fillna("").astype(str).to_numpy()
        side = connectome.neurons["side"].fillna("").astype(str).to_numpy()
        dirs = neuron_directions(connectome, retinotopy, {n for ns in names.values() for n in ns})
        hex_of_type = {ft: np.flatnonzero(self.ntype == ft) for ft in ftypes}   # 721 nodes each
        idx_rows, eyes, nodes, tys = [], [], [], []
        for ft in ftypes:
            fnodes = hex_of_type[ft]
            fu, fv = u[fnodes], v[fnodes]
            for e in ("left", "right"):
                faz, fel = hexal_directions(fu, fv, e)
                for cn in names[ft]:
                    sel = [i for i in np.flatnonzero(cname == cn) if i in dirs and side[i] == e]
                    if not sel:
                        continue
                    naz = np.array([dirs[i][0] for i in sel])
                    nel = np.array([dirs[i][1] for i in sel])
                    d = np.hypot((naz[:, None] - faz[None, :]) * np.cos(np.radians(nel[:, None])),
                                 nel[:, None] - fel[None, :])
                    k = np.argmin(d, axis=1)
                    near = d[np.arange(len(sel)), k] <= EDGE_COLUMNS * OMM_DEG
                    idx_rows += [sel[j] for j in np.flatnonzero(near)]
                    eyes += [0 if e == "left" else 1] * int(near.sum())
                    nodes += list(fnodes[k[near]])
                    tys += [ft] * int(near.sum())
        order = np.argsort(idx_rows, kind="stable")
        self.indices = np.asarray(idx_rows, np.int64)[order]
        self._eye = np.asarray(eyes, np.int64)[order]
        self._node = np.asarray(nodes, np.int64)[order]
        self._ty = np.asarray(tys)[order]

    def save(self, path: str) -> None:
        """The mapping and calibration (no connectome needed to load it)."""
        np.savez_compressed(path, model=np.array(self.model), indices=self.indices, eye=self._eye,
                            node=self._node, scale=self._scale)

    # ------------------------------------------------------------ flyvis
    def _input(self, lum_left, lum_right):
        """Both eyes' column brightness -> flyvis's input (every photoreceptor
        type of a column sees the same), built on flyvis's device in one copy."""
        torch = self.torch
        dev = self._params_device()
        if getattr(self, "_inp_t", None) is None or self._inp_t.device != dev:
            self._inp_t = torch.as_tensor(self.inp.ravel(), dtype=torch.long, device=dev)
        L = torch.as_tensor(np.stack([lum_left, lum_right]).astype(np.float32), device=dev)
        x = torch.zeros(2, len(self.ntype), device=dev)
        x[:, self._inp_t] = L.repeat(1, self.inp.shape[0])
        return x

    def _params_device(self):
        return next(self.net.parameters()).device

    def _step(self, x):
        with self.torch.no_grad():
            self._state = self.net._next_state(self._params, self._state, x, DT)
        return self._state.nodes.activity

    def _graph_step(self, x):
        """One step as a captured CUDA graph (on a GPU, FLY_EYE_GRAPH not 0):
        flyvis's step is a few hundred tiny kernels, launch-bound (Jetson Orin:
        2 ms a step eager), and fewer, shorter GPU bursts also delay the
        brain's own GPU blocks less. Same arithmetic as _step; falls back to
        it if capture fails."""
        torch = self.torch
        g = getattr(self, "_graph", None)
        if g is None:
            if (x.device.type != "cuda" or os.environ.get("FLY_EYE_GRAPH", "1") == "0"
                    or getattr(self, "_graph_failed", False)):
                return self._step(x)
            try:
                self._graph = g = self._capture(x)
            except Exception as ex:                                     # eager from now on
                print("flyvis_eye: CUDA graph capture failed, eager steps:", ex)
                self._graph_failed = True
                return self._step(x)
        xs, nodes, graph = g[:3]
        xs.copy_(x)
        graph.replay()
        return nodes["activity"]

    def _capture(self, x):
        torch = self.torch
        AD = type(self._state)
        nodes = {k: v.clone() for k, v in self._state.nodes.items()}
        edges = {k: v.clone() for k, v in self._state.edges.items()}
        saved = ({k: v.clone() for k, v in nodes.items()}, {k: v.clone() for k, v in edges.items()})
        xs = x.clone()
        # flyvis's dynamics build torch.tensor(dt) every step: from a Python
        # float that is a host-to-device copy, which a graph cannot hold; from
        # a tensor already on the GPU it is a device copy
        dt = torch.tensor(DT, dtype=torch.float32, device=x.device)

        def body():
            st = self.net._state_api(AD(nodes=AD(**nodes), edges=AD(**edges)))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)             # "copy construct from a tensor"
                new = self.net._next_state(self._params, st, xs, dt)
            for k in nodes:
                nodes[k].copy_(new.nodes[k])
            for k in edges:
                edges[k].copy_(new.edges[k])

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.no_grad(), torch.cuda.stream(side):
            for _ in range(3):                                          # warm-up (allocations)
                body()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(graph):
            body()
        for d, sv in zip((nodes, edges), saved):                         # undo the warm-up
            for k in d:
                d[k].copy_(sv[k])
        with torch.no_grad():
            self._state = self.net._state_api(AD(nodes=AD(**nodes), edges=AD(**edges)))
        # everything the graph reads must outlive it (a freed dt was reused
        # by the next allocation: the graph then read garbage)
        return xs, nodes, graph, edges, dt, body

    def _calibrate(self):
        """Per flyvis cell type, the 99th percentile of its response above the
        grey level to 0.6 s of gratings (horizontal, then vertical motion) with
        grey gaps (on / off flashes)."""
        st = self._state
        devs = []
        for k in range(60):
            t = k * DT
            frames = []
            for e in ("left", "right"):
                az, el = self.eye_dir[e]
                pos = az if k < 30 else el
                frames.append(0.5 + 0.5 * np.sign(np.sin(2 * np.pi * (pos - 60.0 * t) / 30.0)) * (1 if k % 20 < 18 else 0))
            a = self._step(self._input(*frames)).detach().cpu().numpy()
            devs.append(np.maximum(a - self._a0, 0))
        dev = np.concatenate(devs, axis=0)                                       # (frames*2, nodes)
        for ft in np.unique(self.ntype):
            m = self.ntype == ft
            s = float(np.percentile(dev[:, m], 99))
            self._scale[m] = max(s, 1e-3)
        self._state = st
        self._rc = None

    def _to_rates(self, act, gathered: bool = False):
        """flyvis activity (2, nodes), or already gathered per mapped neuron
        (see()), -> the mapped neurons' rates (read-only float64)."""
        if getattr(self, "_rc", None) is None:
            self._rc = (self._a0[self._eye, self._node], 1.0 / self._scale[self._node])
        a = act if gathered else act[self._eye, self._node]
        r = RATE_MAX * np.clip((a - self._rc[0]) * self._rc[1], 0.0, 1.5)
        r = r.astype(np.float64)
        r.flags.writeable = False
        return r

    def _sample(self, img):
        """Each eye's columns' brightness: a Gaussian of the acceptance angle
        around each column's direction (wrapping in azimuth), as one sparse
        matrix per eye (built once; a blur of the whole panorama every step
        cost more than flyvis itself)."""
        if getattr(self, "_S", None) is None:
            self._S = {e: sampling_matrix(*self._px[e], self.sigma_px) for e in ("left", "right")}
        flat = np.asarray(img, np.float32).ravel()
        return [self._S[e] @ flat for e in ("left", "right")]

    # ------------------------------------------------------------ encoder
    def rates_hz(self, t_ms: float, stim=None) -> np.ndarray:
        if self._t_ms is None or t_ms < self._t_ms:
            self._t_ms = t_ms                                                   # (a new day: the clock restarts)
        n = int((t_ms - self._t_ms) // (DT * 1000.0))
        if n <= 0:
            return self._rates
        fid, img = self.source(t_ms)
        self.see(img, min(n, 50))                                               # catch up, at most 0.5 s
        self._t_ms += n * DT * 1000.0
        return self._rates

    def see(self, img, n: int = 1) -> np.ndarray:
        """n flyvis steps (of DT) looking at the panorama img; the new rates."""
        lum = self._sample(img)
        m = np.array([float(l.mean()) for l in lum])
        if self._eye_mean is None:
            self._eye_mean = m
        else:
            self._eye_mean = self._eye_mean + (m - self._eye_mean) * (1.0 - np.exp(-n * DT / ADAPT_S))
        x = self._input(*[np.clip(0.5 * l / max(mu, 0.02), 0.0, 1.0) for l, mu in zip(lum, self._eye_mean)])
        t0 = time.perf_counter()
        for _ in range(n):
            act = self._graph_step(x)
        if getattr(self, "_gather", None) is None or self._gather[0].device != act.device:
            self._gather = (self.torch.as_tensor(self._eye, device=act.device),
                            self.torch.as_tensor(self._node, device=act.device))
        act = act[self._gather[0], self._gather[1]].detach().cpu().numpy()
        if self.baseline_s:
            # adaptation: a still scene fades to rest (the fly's visual
            # neurons adapt within seconds), motion and change still drive
            if self._base is None:
                self._base = self._a0[self._eye, self._node].astype(np.float32)
            rel = act - self._base
            self._base += (act - self._base) * np.float32(1.0 - np.exp(-n * DT / self.baseline_s))
            act = rel + self._rc[0]
        self.step_ms_total += 1e3 * (time.perf_counter() - t0)
        self.steps += n
        self._rates = self._to_rates(act, gathered=True)
        self.last = {"mean_hz": round(float(self._rates.mean()), 1), "max_hz": round(float(self._rates.max()), 1)}
        return self._rates

    def state(self, t_ms: float) -> dict:
        return {"kind": "flyvis", "active": True, **self.last}

    @property
    def provenance(self) -> dict:
        return {"drives": {k: v for k, v in self.n_by_type.items() if v},
                "source": "flyvis (Lappalainen et al. 2024), model %s; robot/flyvis_eye.py" % MODEL}
