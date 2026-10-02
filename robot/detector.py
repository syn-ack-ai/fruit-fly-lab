"""
The camera's person detector on the Jetson's GPU: a YOLOX model (Megvii,
Apache-2.0; COCO-trained, class 0 = person) run by TensorRT through
native/libtrt_detect.so. It returns what robot/head.py's Hailo detector
returned on the Pi: the largest person as fractions of the frame
{cx, cy, w, h, score}, so the rest of the head (the head-and-shoulders target,
its azimuth, elevation and angular size) is unchanged. The same inference
also gives the most confident toy (TOYS: a ball, a frisbee, a teddy bear;
detect_all), which Milo pursues through LC10a like a person.

The engine blocks its thread while the GPU works (no GIL held, no spinning
core). It runs on a lowest-priority CUDA stream, which puts the brain's
kernels first only within one process; in robot/head.py's camera process the
GPU time-slices the two (measured on the Orin: the brain loop's timing is
unchanged with the detector at 10 Hz).

Models are not in the repository (deploy/jetson/README.md):
    curl -LO https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.onnx
    python -m robot.detector --build ~/milo/models/yolox_tiny.onnx      # -> yolox_tiny.engine (FP16)
    python -m robot.detector --engine ~/milo/models/yolox_tiny.engine --image some.jpg

C. APPROXIMATIONS: YOLOX's own preprocessing (BGR, 0-255, the image resized
into the top-left of a square padded with 114) and decoding (strides 8, 16,
32; score = objectness x class); a person needs score >= 0.45; overlapping
boxes are merged by non-maximum suppression (IoU 0.45).
"""
from __future__ import annotations

import ctypes as C
import os
import shutil
import subprocess
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.environ.get("FLY_TRT_LIB", os.path.join(HERE, "..", "native", "libtrt_detect.so"))
TRTEXEC = shutil.which("trtexec") or "/usr/src/tensorrt/bin/trtexec"
DEFAULT_ENGINE = os.path.expanduser("~/milo/models/yolox_tiny.engine")
PERSON = 0
# toys Milo may play with (COCO classes YOLOX knows): pursued like a person
# through LC10a (robot/head.ObjectEncoder)
TOYS = {32: "ball", 29: "frisbee", 77: "teddy bear"}
SCORE_MIN = 0.45
TOY_SCORE_MIN = 0.35
NMS_IOU = 0.45
STRIDES = (8, 16, 32)


class TRTEngine:
    """A TensorRT engine with float32 inputs and outputs (native/trt_detect.cpp).
    .inputs / .outputs map tensor names to numpy views of pinned host buffers,
    written and read in place; with one of each, .input / .output are those."""

    def __init__(self, engine_path: str, lib_path: str = LIB):
        lib = C.CDLL(os.path.abspath(lib_path))
        lib.trt_open.restype = C.c_void_p
        lib.trt_open.argtypes = [C.c_char_p, C.c_char_p, C.c_int]
        lib.trt_count.restype = C.c_int
        lib.trt_count.argtypes = [C.c_void_p]
        lib.trt_info.restype = C.c_int
        lib.trt_info.argtypes = [C.c_void_p, C.c_int, C.c_char_p, C.c_int, C.POINTER(C.c_int),
                                 C.POINTER(C.c_int64), C.c_int]
        lib.trt_host.restype = C.POINTER(C.c_float)
        lib.trt_host.argtypes = [C.c_void_p, C.c_int]
        lib.trt_run.restype = C.c_int
        lib.trt_run.argtypes = [C.c_void_p]
        lib.trt_close.restype = None
        lib.trt_close.argtypes = [C.c_void_p]
        err = C.create_string_buffer(512)
        h = lib.trt_open(os.fsencode(engine_path), err, len(err))
        if not h:
            raise RuntimeError(f"TensorRT engine {engine_path}: {err.value.decode(errors='replace')}")
        self._lib, self._h = lib, h
        self.inputs, self.outputs = {}, {}
        for i in range(lib.trt_count(h)):
            name, is_in, d = C.create_string_buffer(256), C.c_int(), (C.c_int64 * 8)()
            n = lib.trt_info(h, i, name, len(name), C.byref(is_in), d, 8)
            shape = tuple(int(d[k]) for k in range(n))
            arr = np.ctypeslib.as_array(lib.trt_host(h, i), shape=shape)
            (self.inputs if is_in.value else self.outputs)[name.value.decode()] = arr
        self.input = next(iter(self.inputs.values())) if len(self.inputs) == 1 else None
        self.output = next(iter(self.outputs.values())) if len(self.outputs) == 1 else None
        self.in_shape = self.input.shape if self.input is not None else None
        self.out_shape = self.output.shape if self.output is not None else None

    def run(self):
        """Inputs -> outputs (the GIL is released while the GPU works). Returns
        .output with one output, else the .outputs dict."""
        if self._h is None:
            raise RuntimeError("engine closed")
        e = self._lib.trt_run(self._h)
        if e:
            raise RuntimeError(f"TensorRT run failed (error {e})")
        return self.output if self.output is not None else self.outputs

    def close(self) -> None:
        if self._h is not None:
            self.input = self.output = None
            self.inputs, self.outputs = {}, {}
            self._lib.trt_close(self._h)
            self._h = None


def _grids(h: int, w: int):
    g, s = [], []
    for st in STRIDES:
        yv, xv = np.mgrid[0:h // st, 0:w // st]
        g.append(np.stack((xv, yv), -1).reshape(-1, 2))
        s.append(np.full((g[-1].shape[0], 1), st))
    return np.concatenate(g).astype(np.float32), np.concatenate(s).astype(np.float32)


def nms(boxes: np.ndarray, scores: np.ndarray, iou: float = NMS_IOU) -> list:
    """Indices kept by greedy non-maximum suppression (boxes x0, y0, x1, y1)."""
    order = np.argsort(-scores, kind="stable")
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        rest = order[1:]
        xx0 = np.maximum(boxes[i, 0], boxes[rest, 0]); yy0 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx1 = np.minimum(boxes[i, 2], boxes[rest, 2]); yy1 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx1 - xx0, 0, None) * np.clip(yy1 - yy0, 0, None)
        order = rest[inter / np.maximum(area[i] + area[rest] - inter, 1e-9) <= iou]
    return keep


def _fractions(p: dict, hw) -> dict:
    h, w = hw
    return {"cx": (p["x0"] + p["x1"]) / (2 * w), "cy": (p["y0"] + p["y1"]) / (2 * h),
            "w": (p["x1"] - p["x0"]) / w, "h": (p["y1"] - p["y0"]) / h, "score": p["score"],
            "area": (p["x1"] - p["x0"]) * (p["y1"] - p["y0"]) / (w * h)}


def _largest(objs: list, hw) -> dict | None:
    if not objs:
        return None
    return _fractions(max(objs, key=lambda p: (p["x1"] - p["x0"]) * (p["y1"] - p["y0"])), hw)


class YoloxPersons:
    """People in a BGR frame, from a YOLOX engine (input 1 x 3 x H x W)."""

    def __init__(self, engine_path: str = DEFAULT_ENGINE, score_min: float = SCORE_MIN,
                 engine: TRTEngine | None = None):
        self.eng = engine or TRTEngine(engine_path)
        try:
            _, _, self.H, self.W = self.eng.in_shape
            self.grid, self.stride = _grids(self.H, self.W)
            if len(self.eng.out_shape) != 3 or self.eng.out_shape[-2] != self.grid.shape[0]:
                raise RuntimeError(f"unexpected YOLOX output {self.eng.out_shape} for input {self.eng.in_shape}")
        except Exception:
            self.eng.close()
            raise
        self.score_min = float(score_min)
        self.ms = 0.0                                   # last inference time (ms)
        self.bgr = True                                 # detect() takes BGR frames (robot/head.py)

    def preprocess(self, bgr: np.ndarray) -> float:
        """Writes the padded image into the engine's input; returns the scale."""
        import cv2
        h, w = bgr.shape[:2]
        r = min(self.H / h, self.W / w)
        nh, nw = int(h * r), int(w * r)
        img = np.full((self.H, self.W, 3), 114, np.uint8)
        img[:nh, :nw] = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        self.eng.input[0] = img.transpose(2, 0, 1)      # float32, HWC -> CHW
        return r

    def decode(self, out: np.ndarray, r: float, frame_hw, cls: int = PERSON,
               score_min: float | None = None) -> list:
        """Raw output (N x 85) -> objects of one class [{x0, y0, x1, y1 (px), score}]."""
        o = out.reshape(-1, out.shape[-1])
        score = o[:, 4] * o[:, 5 + cls]
        m = score >= (self.score_min if score_min is None else score_min)
        if not m.any():
            return []
        o, score = o[m], score[m]
        xy = (o[:, :2] + self.grid[m]) * self.stride[m]
        wh = np.exp(o[:, 2:4]) * self.stride[m]
        b = np.concatenate([xy - wh / 2, xy + wh / 2], 1) / r
        h, w = frame_hw
        b[:, [0, 2]] = np.clip(b[:, [0, 2]], 0, w)
        b[:, [1, 3]] = np.clip(b[:, [1, 3]], 0, h)
        big = (b[:, 2] - b[:, 0] > 1.0) & (b[:, 3] - b[:, 1] > 1.0)   # not a sliver at an edge
        b, score = b[big], score[big]
        return [{"x0": float(b[i, 0]), "y0": float(b[i, 1]), "x1": float(b[i, 2]),
                 "y1": float(b[i, 3]), "score": float(score[i])} for i in nms(b, score)]

    def people(self, bgr: np.ndarray) -> list:
        r = self.preprocess(bgr)
        t0 = time.perf_counter()
        out = self.eng.run()
        self.ms = 1e3 * (time.perf_counter() - t0)
        return self.decode(out, r, bgr.shape[:2])

    def detect(self, bgr: np.ndarray) -> dict | None:
        """The largest person as fractions of the frame (robot/head.py's format)."""
        return self.detect_all(bgr)["person"]

    def detect_all(self, bgr: np.ndarray) -> dict:
        """One inference: {"person": the largest person, "toy": the most
        confident toy (with "name")}, each as fractions of the frame or None."""
        r = self.preprocess(bgr)
        t0 = time.perf_counter()
        out = self.eng.run()
        self.ms = 1e3 * (time.perf_counter() - t0)
        hw = bgr.shape[:2]
        person = _largest(self.decode(out, r, hw), hw)
        toy = None
        for c, name in TOYS.items():
            for d in self.decode(out, r, hw, c, TOY_SCORE_MIN):
                if toy is None or d["score"] > toy[0]["score"]:
                    toy = (d, name)
        if toy is not None:
            toy = dict(_fractions(toy[0], hw), name=toy[1])
        return {"person": person, "toy": toy}

    def close(self) -> None:
        self.eng.close()


def build_engine(onnx_path: str, out_path: str | None = None, fp16: bool = True) -> str:
    """ONNX -> a TensorRT engine for this GPU (a few minutes on the Orin): with
    trtexec (JetPack) or else TensorRT's Python API (the pip wheel). Engines are
    specific to the GPU and the TensorRT version."""
    out_path = out_path or os.path.splitext(onnx_path)[0] + ".engine"
    if os.path.exists(TRTEXEC):
        cmd = [TRTEXEC, f"--onnx={onnx_path}", f"--saveEngine={out_path}.tmp", "--skipInference"]
        if fp16:
            cmd.append("--fp16")
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    else:
        import tensorrt as trt
        log = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(log)
        major = int(trt.__version__.split(".")[0])
        net = builder.create_network(0 if major >= 10 else
                                     1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(net, log)
        if not parser.parse_from_file(onnx_path):
            raise RuntimeError("; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
        for i in range(net.num_inputs):
            if any(d < 0 for d in net.get_input(i).shape):
                raise RuntimeError(f"{onnx_path}: input {net.get_input(i).name} has a dynamic shape; "
                                   "fix its size in the ONNX file (or use trtexec --shapes)")
        cfg = builder.create_builder_config()
        if fp16 and hasattr(trt.BuilderFlag, "FP16"):
            cfg.set_flag(trt.BuilderFlag.FP16)
        blob = builder.build_serialized_network(net, cfg)
        if blob is None:
            raise RuntimeError("TensorRT could not build the engine")
        with open(out_path + ".tmp", "wb") as fh:
            fh.write(blob)
    os.replace(out_path + ".tmp", out_path)
    return out_path


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", metavar="ONNX", help="build an FP16 engine next to the ONNX file")
    ap.add_argument("--engine", default=DEFAULT_ENGINE)
    ap.add_argument("--image", help="detect people in an image file")
    ap.add_argument("--camera", help="detect people and toys from a V4L2 camera (e.g. /dev/video0)")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--bench", type=int, default=0, help="time N inferences")
    a = ap.parse_args()
    if a.build:
        print("built", build_engine(a.build))
        return
    import cv2
    det = YoloxPersons(a.engine)
    print("engine", a.engine, "input", det.eng.in_shape, "output", det.eng.out_shape)
    try:
        if a.image:
            img = cv2.imread(a.image)
            for p in det.people(img):
                print({k: round(v, 3) for k, v in p.items()})
            print("largest:", det.detect(img), f"({det.ms:.2f} ms)")
            if a.bench:
                ts = []
                for _ in range(a.bench):
                    t0 = time.perf_counter()
                    det.detect(img)
                    ts.append(1e3 * (time.perf_counter() - t0))
                ts = np.array(ts[5:] if len(ts) > 10 else ts)
                print(f"detect() incl. pre/post: mean {ts.mean():.2f} ms, p95 {np.percentile(ts, 95):.2f} ms;"
                      f" inference alone {det.ms:.2f} ms")
        if a.camera:
            cap = cv2.VideoCapture(a.camera, cv2.CAP_V4L2)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
            t0 = time.monotonic()
            while time.monotonic() - t0 < a.seconds:
                ok, frame = cap.read()
                if ok:
                    found = det.detect_all(frame)
                    p, toy = found["person"], found["toy"]
                    print(f"{time.monotonic() - t0:5.1f} s  person "
                          + (f"{p['score']:.2f} at x {p['cx']:.2f}" if p else "-   ")
                          + "  toy " + (f"{toy['name']} {toy['score']:.2f} at x {toy['cx']:.2f}, "
                                        f"{100 * toy['w']:.0f}% wide" if toy else "-")
                          + f"  ({det.ms:.1f} ms)", flush=True)
            cap.release()
    finally:
        det.close()


if __name__ == "__main__":
    main()
