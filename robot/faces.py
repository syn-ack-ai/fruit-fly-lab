"""
Faces through the robot's camera: where they are, whether they face the robot,
and who they are (robot/people.py keeps the people the robot knows).

    frame -> YuNet (faces, 5 landmarks) -> facing the camera?  (landmark geometry)
                                        -> aligned 112x112 crop -> SFace -> 128-d embedding

Models (OpenCV Zoo; not in the repository, ~/milo/models):
    face_detection_yunet_2023mar.onnx      YuNet (Wu et al. 2023), MIT
    face_recognition_sface_2021dec.onnx    SFace (Zhong et al. 2021), Apache-2.0
On a Jetson they run on the GPU through TensorRT (native/libtrt_detect.so, the
same runner as robot/detector.py); elsewhere, or without engines, through
OpenCV's own FaceDetectorYN / FaceRecognizerSF on the CPU. Both give the same
faces and embeddings (tests/test_faces.py; checked on an RTX 3080 Ti).

    python -m robot.faces --build                     # TensorRT engines next to the models
    python -m robot.faces --image photo.jpg           # faces, facing, embeddings
    python -m robot.faces --camera /dev/video0        # 10 s live

C. APPROXIMATIONS: "facing the robot" is the head turned towards the camera,
from the landmarks (the nose between the eyes, and between the eyes and the
mouth), not where the eyes look; a face needs eyes >= MIN_EYES_PX apart for an
embedding good enough to recognise: within ~1.5 m with the Orbit at 320x240
and 53 degrees (a person further away is seen, not recognised).
"""
from __future__ import annotations

import os
import time

import numpy as np

MODELS = os.path.expanduser(os.environ.get("FLY_MODELS", "~/milo/models"))
YUNET = "face_detection_yunet_2023mar"
SFACE = "face_recognition_sface_2021dec"
SCORE_MIN = 0.7            # YuNet face score
NMS_IOU = 0.3              # as OpenCV's FaceDetectorYN default
STRIDES = (8, 16, 32)
MIN_EYES_PX = 14           # eyes this far apart (in the frame) for an embedding
FACING_YAW = 0.30          # |nose offset| / eye distance below this (~30 deg of head turn): facing
FACING_PITCH = (0.25, 0.85)  # nose between the eye line and the mouth line
# SFace's 112x112 template (OpenCV face_recognize.cpp): eyes, nose, mouth corners
TEMPLATE = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                     [41.5493, 92.3655], [70.7299, 92.2041]], np.float64)


def _path(name: str, ext: str) -> str:
    return os.path.join(MODELS, name + ext)


# ------------------------------------------------------------------ geometry
def facing(landmarks: np.ndarray) -> tuple:
    """(yaw, pitch, facing): yaw = the nose's offset along the eye line in eye
    distances (0 frontal, + = towards the image's right), pitch = the nose's
    position from the eye line (0) to the mouth line (1)."""
    lm = np.asarray(landmarks, float).reshape(5, 2)
    e0, e1, nose, m0, m1 = lm
    eye_mid, mouth_mid = (e0 + e1) / 2, (m0 + m1) / 2
    ax = e1 - e0
    d = float(np.hypot(*ax))
    if d < 1e-6:
        return 0.0, 0.5, False
    ux = ax / d
    yaw = float(np.dot(nose - eye_mid, ux)) / d
    down = mouth_mid - eye_mid
    dl = float(np.dot(down, down))
    pitch = float(np.dot(nose - eye_mid, down) / dl) if dl > 1e-6 else 0.5
    ok = abs(yaw) < FACING_YAW and FACING_PITCH[0] < pitch < FACING_PITCH[1]
    return yaw, pitch, bool(ok)


def similarity_transform(src: np.ndarray, dst: np.ndarray = TEMPLATE) -> np.ndarray:
    """The least-squares similarity (rotation, uniform scale, shift) taking src
    points onto dst (Umeyama 1991), as a 2x3 matrix for cv2.warpAffine."""
    src, dst = np.asarray(src, np.float64), np.asarray(dst, np.float64)
    ms, md = src.mean(0), dst.mean(0)
    s0, d0 = src - ms, dst - md
    cov = d0.T @ s0 / len(src)
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[1, 1] = -1
    R = U @ D @ Vt
    var = (s0 ** 2).sum() / len(src)
    scale = float(np.trace(np.diag(S) @ D) / var) if var > 0 else 1.0
    t = md - scale * R @ ms
    return np.hstack([scale * R, t[:, None]])


def align(bgr: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """The 112x112 face crop SFace expects (eyes, nose, mouth on its template)."""
    import cv2
    M = similarity_transform(np.asarray(landmarks, float).reshape(5, 2))
    return cv2.warpAffine(bgr, M, (112, 112), flags=cv2.INTER_LINEAR, borderValue=0)


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    a = a / max(np.linalg.norm(a), 1e-12)
    bn = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-12)
    return bn @ a


# ------------------------------------------------------------------ YuNet
def yunet_decode(outs: dict, pad_hw, score_min: float = SCORE_MIN, nms_iou: float = NMS_IOU) -> np.ndarray:
    """YuNet's 12 outputs (cls/obj/bbox/kps per stride) -> faces as rows
    [x0, y0, w, h, 10 landmark coordinates, score] in the padded input's
    pixels (OpenCV FaceDetectorYN's layout and decoding)."""
    from robot.detector import nms
    H, W = pad_hw
    rows = []
    for st in STRIDES:
        cols = W // st
        cls = np.clip(np.asarray(outs[f"cls_{st}"], np.float32).reshape(-1), 0, 1)
        obj = np.clip(np.asarray(outs[f"obj_{st}"], np.float32).reshape(-1), 0, 1)
        score = np.sqrt(cls * obj)
        m = score >= score_min
        if not m.any():
            continue
        idx = np.flatnonzero(m)
        c, r = (idx % cols).astype(np.float32), (idx // cols).astype(np.float32)
        bb = np.asarray(outs[f"bbox_{st}"], np.float32).reshape(-1, 4)[idx]
        kp = np.asarray(outs[f"kps_{st}"], np.float32).reshape(-1, 10)[idx]
        cx, cy = (c + bb[:, 0]) * st, (r + bb[:, 1]) * st
        w, h = np.exp(bb[:, 2]) * st, np.exp(bb[:, 3]) * st
        lx = (kp[:, 0::2] + c[:, None]) * st
        ly = (kp[:, 1::2] + r[:, None]) * st
        lm = np.stack([lx, ly], -1).reshape(-1, 10)
        rows.append(np.column_stack([cx - w / 2, cy - h / 2, w, h, lm, score[idx]]))
    if not rows:
        return np.zeros((0, 15), np.float32)
    f = np.concatenate(rows)
    xyxy = np.column_stack([f[:, 0], f[:, 1], f[:, 0] + f[:, 2], f[:, 1] + f[:, 3]])
    return f[nms(xyxy, f[:, 14], nms_iou)].astype(np.float32)


class FaceDetector:
    """YuNet: TensorRT engine if there is one (and the runner), else OpenCV."""

    def __init__(self, engine: str | None = None, onnx: str | None = None, backend: str = "auto",
                 score_min: float = SCORE_MIN):
        self.score_min = float(score_min)
        engine = engine or _path(YUNET, ".engine")
        onnx = onnx or _path(YUNET, ".onnx")
        self.trt = None
        if backend in ("auto", "trt") and os.path.exists(engine):
            try:
                from robot.detector import TRTEngine
                self.trt = TRTEngine(engine)
                (self.inp,) = self.trt.inputs.values()
                _, _, self.H, self.W = self.inp.shape
            except Exception:
                if self.trt is not None:
                    self.trt.close()
                if backend == "trt":
                    raise
                self.trt = None
        if self.trt is None:
            if backend == "trt":
                raise RuntimeError(f"no TensorRT engine {engine}")
            import cv2
            self.cv = cv2.FaceDetectorYN.create(onnx, "", (320, 320), self.score_min, NMS_IOU, 5000)
            self._size = None
        self.backend = "trt" if self.trt is not None else "opencv"
        self.ms = 0.0

    def detect(self, bgr: np.ndarray) -> np.ndarray:
        """Faces in a BGR frame: rows [x0, y0, w, h, 10 landmark coordinates,
        score] in frame pixels (FaceDetectorYN's layout)."""
        t0 = time.perf_counter()
        h, w = bgr.shape[:2]
        if self.trt is not None:
            import cv2
            r = min(self.H / h, self.W / w)
            nh, nw = int(round(h * r)), int(round(w * r))
            img = np.zeros((self.H, self.W, 3), np.uint8)
            img[:nh, :nw] = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR) if r != 1 else bgr
            self.inp[0] = img.transpose(2, 0, 1)
            f = yunet_decode(self.trt.run(), (self.H, self.W), self.score_min)
            f[:, :14] /= r
        else:
            if self._size != (w, h):
                self.cv.setInputSize((w, h))
                self._size = (w, h)
            _, f = self.cv.detect(bgr)
            f = np.zeros((0, 15), np.float32) if f is None else f.astype(np.float32)
        self.ms = 1e3 * (time.perf_counter() - t0)
        return f

    def close(self) -> None:
        if self.trt is not None:
            self.inp = None                     # the pinned buffer goes with the engine
            self.trt.close()
            self.trt = None


# ------------------------------------------------------------------ SFace
class FaceEmbedder:
    """SFace: an aligned face -> a 128-d embedding (L2-normalised here)."""

    def __init__(self, engine: str | None = None, onnx: str | None = None, backend: str = "auto"):
        engine = engine or _path(SFACE, ".engine")
        onnx = onnx or _path(SFACE, ".onnx")
        self.trt = None
        if backend in ("auto", "trt") and os.path.exists(engine):
            try:
                from robot.detector import TRTEngine
                self.trt = TRTEngine(engine)
                (self.inp,) = self.trt.inputs.values()
            except Exception:
                if self.trt is not None:
                    self.trt.close()
                if backend == "trt":
                    raise
                self.trt = None
        if self.trt is None:
            if backend == "trt":
                raise RuntimeError(f"no TensorRT engine {engine}")
            import cv2
            self.cv = cv2.FaceRecognizerSF.create(onnx, "")
        self.backend = "trt" if self.trt is not None else "opencv"
        self.ms = 0.0

    def embed_aligned(self, crop: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        if self.trt is not None:
            self.inp[0] = crop[:, :, ::-1].transpose(2, 0, 1)      # BGR -> RGB, 0-255 (as OpenCV)
            e = np.array(self.trt.run(), np.float32).reshape(-1)
        else:
            e = self.cv.feature(crop).reshape(-1).astype(np.float32)
        self.ms = 1e3 * (time.perf_counter() - t0)
        return e / max(float(np.linalg.norm(e)), 1e-12)

    def embed(self, bgr: np.ndarray, face_row: np.ndarray) -> np.ndarray:
        return self.embed_aligned(align(bgr, face_row[4:14]))

    def close(self) -> None:
        if self.trt is not None:
            self.inp = None
            self.trt.close()
            self.trt = None


class Faces:
    """Detector + embedder: faces in a frame as dicts {box (x0, y0, x1, y1),
    landmarks (5x2), score, yaw, pitch, facing, eyes_px, embedding (or None
    when too small)}."""

    def __init__(self, backend: str = "auto", embed: bool = True):
        self.det = FaceDetector(backend=backend)
        try:
            self.emb = FaceEmbedder(backend=backend) if embed else None
        except Exception:
            self.det.close()
            raise

    def __call__(self, bgr: np.ndarray, embed_min_eyes_px: float = MIN_EYES_PX) -> list:
        out = []
        for f in self.det.detect(bgr):
            lm = f[4:14].reshape(5, 2)
            yaw, pitch, ok = facing(lm)
            eyes = float(np.hypot(*(lm[1] - lm[0])))
            e = (self.emb.embed(bgr, f) if self.emb is not None and eyes >= embed_min_eyes_px else None)
            out.append({"box": (float(f[0]), float(f[1]), float(f[0] + f[2]), float(f[1] + f[3])),
                        "landmarks": lm.astype(float), "score": float(f[14]), "yaw": yaw, "pitch": pitch,
                        "facing": ok, "eyes_px": eyes, "embedding": e})
        return out

    def close(self) -> None:
        self.det.close()
        if self.emb is not None:
            self.emb.close()


def build(models: str = MODELS) -> list:
    """TensorRT engines for both models (FP16), next to the ONNX files."""
    from robot.detector import build_engine
    return [build_engine(os.path.join(models, m + ".onnx")) for m in (YUNET, SFACE)]


def check(images: list) -> list:
    """TensorRT against OpenCV's own YuNet and SFace on the same images (a new
    GPU, a new TensorRT): per image the largest box/landmark difference (px) on
    the same padded input and the embeddings' cosine. Returns those rows."""
    import cv2
    det_t, emb_t = FaceDetector(backend="trt"), FaceEmbedder(backend="trt")
    rec = cv2.FaceRecognizerSF.create(_path(SFACE, ".onnx"), "")
    rows = []
    for path in images:
        img = cv2.imread(path)
        h, w = img.shape[:2]
        r = min(det_t.H / h, det_t.W / w)
        pad = np.zeros((det_t.H, det_t.W, 3), np.uint8)
        nh, nw = int(round(h * r)), int(round(w * r))
        pad[:nh, :nw] = cv2.resize(img, (nw, nh))
        cvd = cv2.FaceDetectorYN.create(_path(YUNET, ".onnx"), "", (det_t.W, det_t.H), SCORE_MIN, NMS_IOU, 5000)
        _, fo = cvd.detect(pad)
        ft = det_t.detect(pad)
        fo = np.zeros((0, 15), np.float32) if fo is None else fo[np.argsort(-fo[:, 14])]
        ft = ft[np.argsort(-ft[:, 14])]
        k = min(len(fo), len(ft))
        row = {"image": path, "faces_opencv": len(fo), "faces_trt": len(ft),
               "max_px": float(np.abs(fo[:k, :14] - ft[:k, :14]).max()) if k else None, "cosine": None}
        if k:
            a = rec.alignCrop(pad, fo[0])
            e_cv = rec.feature(a).reshape(-1)
            row["cosine"] = float(emb_t.embed_aligned(align(pad, fo[0][4:14])) @ (e_cv / np.linalg.norm(e_cv)))
        rows.append(row)
    det_t.close()
    emb_t.close()
    return rows


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="build the TensorRT engines")
    ap.add_argument("--backend", default="auto", choices=("auto", "trt", "opencv"))
    ap.add_argument("--image")
    ap.add_argument("--camera")
    ap.add_argument("--check", nargs="+", metavar="IMAGE", help="TensorRT vs OpenCV on these images")
    a = ap.parse_args()
    if a.build:
        print("built", build())
        return
    if a.check:
        for r in check(a.check):
            print(r)
        return
    import cv2
    fx = Faces(a.backend)
    print("backends: detector", fx.det.backend, "embedder", fx.emb.backend)
    try:
        if a.image:
            img = cv2.imread(a.image)
            for f in fx(img):
                print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in f.items()
                       if k not in ("landmarks", "embedding")}, "embedding" if f["embedding"] is not None else "")
            print(f"detect {fx.det.ms:.2f} ms, embed {fx.emb.ms:.2f} ms")
        if a.camera:
            cap = cv2.VideoCapture(a.camera, cv2.CAP_V4L2)
            t0 = time.monotonic()
            while time.monotonic() - t0 < 10:
                ok, frame = cap.read()
                if ok:
                    fs = fx(frame)
                    print([(round(f["yaw"], 2), f["facing"], round(f["eyes_px"])) for f in fs],
                          f"{fx.det.ms:.1f} ms", flush=True)
            cap.release()
    finally:
        fx.close()


if __name__ == "__main__":
    main()
