"""
What is it, and have I seen it? An image-embedding model on the camera, a
memory of what Milo has seen, and curiosity about what is new.

    camera frame -> four views: the whole frame and its left / middle / right
    halves (overlapping) -> SigLIP 2 base (Google, Apache-2.0; image encoder
    as a TensorRT engine, ~20 ms for the four on the Orin) -> unit vectors
    (robot/head.py's camera process, SEE_PERIOD_S; shared memory)

    brain side (Curiosity, here):
      familiarity  a memory of seen views (vectors with a familiarity 0..1):
                   a view like a remembered one (cosine >= MATCH) makes that
                   memory more familiar (habituation, HABITUATE per look);
                   an unlike one is remembered anew
      novelty      per view, 1 - the familiarity of what it resembles; it
                   fades as Milo keeps looking (seconds), and comes back for
                   something it has not seen
      curiosity    the most novel side, if novel enough, is a target for the
                   pursuit pathway (LC10a, robot/head.ObjectEncoder "novel"):
                   Milo turns to and approaches what is new -- after people
                   and toys
      what         zero-shot labels (text embeddings of LABELS, made once on
                   a bigger machine: `python -m robot.seeing --export DIR`);
                   something new and recognisable is told to the personality
                   ("you notice something new on the left: a cat")

Measured (2026-10-01): the same view twice ~0.96 cosine; the same view with a
knife held up ~0.74; labels: the desk "a chair, a desk, a person", the knife
frame "a knife" (sigmoid 0.14-0.17).

C. APPROXIMATIONS: robot engineering, not a fly circuit. Novelty -> LC10a is
a design choice: the fly's own visual novelty responses (e.g. habituation of
LC neurons) are not modelled. Familiarity is per view, not per object.
"""
from __future__ import annotations

import os
import time

import numpy as np

MODELS = os.path.expanduser(os.environ.get("FLY_MODELS", "~/milo/models"))
ENGINE = os.path.join(MODELS, "siglip2_b16_vision.engine")
LABELS = os.path.join(MODELS, "siglip2_b16_labels.npz")
HF_NAME = "google/siglip2-base-patch16-224"
DIM = 768
SIZE = 224
SEE_PERIOD_S = 0.5            # a look this often (camera process)
VIEWS = ("whole", "left", "middle", "right")
VIEW_AZ_FRAC = {"left": -0.25, "middle": 0.0, "right": 0.25}   # view centres, x the camera's HFOV

MATCH = 0.90                  # cosine: "the same thing / view"
LIKE = 0.75                   # below this, nothing alike: wholly novel
HABITUATE = 0.06              # familiarity gained per look (2 a second: ~5 s of interest)
NEW_FAMILIARITY = 0.1
CAPACITY = 4096
NOVEL_MIN = 0.55              # novelty of a side that draws curiosity
LABEL_MIN = 0.01              # sigmoid score of a label worth naming (a desk ~0.01, a knife ~0.15)
TELL_EVERY_S = 15.0           # at most one "something new" for the personality this often


def views(rgb: np.ndarray) -> list:
    """The four views of an RGB frame (H x W x 3 uint8): whole, left, middle,
    right (halves of the width, overlapping by a quarter)."""
    h, w = rgb.shape[:2]
    half = w // 2
    return [rgb, rgb[:, :half], rgb[:, w // 4:w // 4 + half], rgb[:, w - half:]]


def preprocess(rgb: np.ndarray) -> np.ndarray:
    """SigLIP's preprocessing: resize to 224 x 224 (bilinear), [-1, 1], CHW."""
    import cv2
    a = cv2.resize(rgb, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    return ((a - 0.5) / 0.5).transpose(2, 0, 1)


class Seer:
    """The image encoder (TensorRT engine with input 4 x 3 x 224 x 224)."""

    def __init__(self, engine: str = ENGINE):
        from robot.detector import TRTEngine
        self.eng = TRTEngine(engine)
        if tuple(self.eng.in_shape) != (len(VIEWS), 3, SIZE, SIZE) or self.eng.out_shape[-1] != DIM:
            shape = (self.eng.in_shape, self.eng.out_shape)
            self.eng.close()
            raise RuntimeError(f"unexpected SigLIP engine shapes {shape}")
        self.ms = 0.0

    def see(self, rgb: np.ndarray) -> np.ndarray:
        """The four views' embeddings (4 x 768, unit length)."""
        self.eng.input[:] = np.stack([preprocess(v) for v in views(rgb)])
        t0 = time.perf_counter()
        out = self.eng.run()
        self.ms = 1e3 * (time.perf_counter() - t0)
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-6)

    def close(self) -> None:
        self.eng.close()


class Labels:
    """Zero-shot labels: text embeddings with SigLIP's sigmoid scale and bias."""

    def __init__(self, path: str = LABELS):
        d = np.load(path)
        self.names = [str(x) for x in d["labels"]]
        self.text = d["text"].astype(np.float32)
        self.scale, self.bias = float(d["logit_scale"]), float(d["logit_bias"])

    def top(self, e: np.ndarray) -> tuple:
        """(label, score) for one embedding."""
        logit = self.text @ e * self.scale + self.bias
        i = int(np.argmax(logit))
        return self.names[i], float(1.0 / (1.0 + np.exp(-logit[i])))


class Memory:
    """Seen views and how familiar each is."""

    def __init__(self, capacity: int = CAPACITY):
        self.capacity = capacity
        self.vec = np.zeros((0, DIM), np.float32)
        self.fam = np.zeros(0, np.float32)
        self.last = np.zeros(0, np.float64)          # last time seen (for forgetting the oldest)

    def __len__(self):
        return len(self.fam)

    def novelty(self, e: np.ndarray) -> float:
        """1 - the familiarity of what this looks like (weighted by how alike)."""
        if not len(self):
            return 1.0
        sim = self.vec @ e
        alike = np.clip((sim - LIKE) / (MATCH - LIKE), 0.0, 1.0)
        return float(1.0 - np.max(alike * self.fam))

    def learn(self, e: np.ndarray, t: float, done: set | None = None) -> None:
        """One look: habituate to what it resembles, or remember it anew.
        `done`: memories already habituated in this look (several views of
        one look can resemble the same memory: it counts once)."""
        if len(self):
            sim = self.vec @ e
            j = int(np.argmax(sim))
            if sim[j] >= MATCH:
                if done is None or j not in done:
                    self.fam[j] += HABITUATE * (1.0 - self.fam[j])
                    v = 0.95 * self.vec[j] + 0.05 * e
                    self.vec[j] = v / np.linalg.norm(v)
                    self.last[j] = t
                    if done is not None:
                        done.add(j)
                return
        if len(self) >= self.capacity:
            k = int(np.argmin(self.last))            # forget the longest unseen
            self.vec[k], self.fam[k], self.last[k] = e, NEW_FAMILIARITY, t
            return
        self.vec = np.vstack([self.vec, e[None].astype(np.float32)])
        self.fam = np.append(self.fam, np.float32(NEW_FAMILIARITY))
        self.last = np.append(self.last, t)

    def save(self, path: str) -> None:
        tmp = path + ".tmp.npz"
        np.savez(tmp, vec=self.vec.astype(np.float16), fam=self.fam, last=self.last)
        os.replace(tmp, path)

    def load(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        d = np.load(path)
        self.vec = d["vec"].astype(np.float32)
        self.fam, self.last = d["fam"].astype(np.float32), d["last"].astype(np.float64)
        return True


class Curiosity:
    """The brain side: novelty of what the camera shows, a curiosity target
    for the pursuit pathway, and news for the personality."""

    def __init__(self, hfov_deg: float, labels: Labels | None = None, state_path: str | None = None):
        self.hfov = float(hfov_deg)
        self.labels = labels
        self.mem = Memory()
        self.path = os.path.join(state_path, "seen.npz") if state_path else None
        if self.path:
            self.mem.load(self.path)
        self._seq = None
        self._told_t = -1e9
        self.target = None               # {"az", "el", "half", "novelty", "what"} or None
        self.last = {}
        self.stats = {"looks": 0, "novel_looks": 0, "told": 0}

    def step(self, t: float, seen: dict | None, personality=None) -> dict | None:
        """seen = {"seq", "emb" (4 x 768), "pan", "tilt"} from the camera (or
        None). Updates the novelty and the target on a new look."""
        if seen is None or seen.get("emb") is None:
            self.target = None
            return None
        if seen["seq"] == self._seq:
            return self.target
        self._seq = seen["seq"]
        emb = np.asarray(seen["emb"], np.float32)
        nov = {v: self.mem.novelty(emb[i]) for i, v in enumerate(VIEWS)}
        done = set()
        for i in range(len(VIEWS)):
            self.mem.learn(emb[i], t, done)
        self.stats["looks"] += 1
        side = max(VIEW_AZ_FRAC, key=lambda v: nov[v])
        what, score = (None, 0.0)
        if self.labels is not None:
            what, score = self.labels.top(emb[VIEWS.index(side)])
        self.last = {"novelty": {k: round(v, 2) for k, v in nov.items()}, "side": side,
                     "what": what if score >= LABEL_MIN else None, "memory": len(self.mem)}
        if nov[side] >= NOVEL_MIN:
            self.stats["novel_looks"] += 1
            self.target = {"az": float(seen.get("pan", 0.0)) + VIEW_AZ_FRAC[side] * self.hfov,
                           "el": float(seen.get("el", 0.0)), "half": 8.0, "novelty": nov[side],
                           "what": self.last["what"]}
            if (personality is not None and self.last["what"] and nov[side] >= 0.8
                    and t - self._told_t >= TELL_EVERY_S):
                self._told_t = t
                self.stats["told"] += 1
                where = {"left": " on the left", "right": " on the right"}.get(side, "")
                personality.event(t, f"you notice something new{where}: {self.last['what']}")
        else:
            self.target = None
        return self.target

    def save(self) -> None:
        if self.path:
            self.mem.save(self.path)


def export(out_dir: str, labels: list | None = None) -> None:
    """On a machine with transformers and onnx (not the robot): the image
    encoder as ONNX (batch 4) and the label text embeddings. Then on the
    Jetson: python -m robot.detector --build DIR/siglip2_b16_vision.onnx"""
    import torch
    from transformers import AutoModel, AutoProcessor
    labels = labels or DEFAULT_LABELS
    m = AutoModel.from_pretrained(HF_NAME).eval()
    proc = AutoProcessor.from_pretrained(HF_NAME)

    class Vision(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, pixels):
            f = self.m.get_image_features(pixel_values=pixels)
            f = getattr(f, "pooler_output", f)
            return f / f.norm(dim=-1, keepdim=True)
    os.makedirs(out_dir, exist_ok=True)
    torch.onnx.export(Vision(m), torch.zeros(len(VIEWS), 3, SIZE, SIZE),
                      os.path.join(out_dir, "siglip2_b16_vision.onnx"), input_names=["pixels"],
                      output_names=["embeds"], opset_version=17, dynamo=False)
    with torch.no_grad():
        t = proc(text=[f"a photo of {x}." for x in labels], padding="max_length", max_length=64,
                 return_tensors="pt")
        tf = m.get_text_features(**t)
        tf = getattr(tf, "pooler_output", tf)
        tf = tf / tf.norm(dim=-1, keepdim=True)
    np.savez(os.path.join(out_dir, "siglip2_b16_labels.npz"), labels=np.array(labels),
             text=tf.numpy().astype(np.float32), logit_scale=float(m.logit_scale.exp()),
             logit_bias=float(m.logit_bias), model=HF_NAME)


DEFAULT_LABELS = [
    "a person", "a person's face", "a hand", "a foot", "a cat", "a dog", "a ball", "a toy", "a teddy bear",
    "a shoe", "a sock", "a chair", "a table", "a sofa", "a bed", "a door", "a wall", "a floor", "a rug",
    "a plant", "a cup", "a bottle", "a bag", "a cardboard box", "a book", "a laptop", "a phone",
    "a television", "a cable", "stairs", "a kitchen", "a bathroom", "a window", "a robot vacuum",
    "a knife", "fire", "water on the floor", "a desk", "a lamp", "a basket", "clothes", "a toy car",
    "a remote control", "a pillow", "a guitar", "food", "a bicycle", "a mirror"]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", metavar="DIR", help="write the ONNX image encoder and label embeddings")
    ap.add_argument("--camera", help="watch novelty live from a V4L2 camera, e.g. /dev/video0")
    ap.add_argument("--seconds", type=float, default=60.0)
    a = ap.parse_args()
    if a.export:
        export(a.export)
        print("wrote", a.export)
        return
    if a.camera:
        import cv2
        seer, labels, cur = Seer(), Labels(), Curiosity(53.0, Labels())
        cap = cv2.VideoCapture(a.camera, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        t0, k = time.monotonic(), 0
        try:
            while time.monotonic() - t0 < a.seconds:
                for _ in range(3):
                    ok, frame = cap.read()
                if not ok:
                    continue
                emb = seer.see(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                k += 1
                t = time.monotonic() - t0
                tgt = cur.step(t, {"seq": k, "emb": emb})
                L = cur.last
                print(f"{t:5.1f} s  novelty " + " ".join(f"{v[0]}{L['novelty'][v]:.2f}" for v in VIEWS)
                      + f"  most new: {L['side']:6} {L['what'] or '-':16} memory {L['memory']:4}"
                      + (f"  -> curious, look {tgt['az']:+.0f} deg" if tgt else "") + f"  ({seer.ms:.0f} ms)",
                      flush=True)
                time.sleep(max(0.0, SEE_PERIOD_S - 0.1))
        finally:
            cap.release()
            seer.close()


if __name__ == "__main__":
    main()
