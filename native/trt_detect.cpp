/*
 * A TensorRT engine runner for the robot's camera detector (robot/detector.py),
 * called through ctypes so the inference never holds Python's GIL.
 *
 * One engine with one float32 input and one float32 output, static shapes
 * (an ONNX model built with trtexec --fp16; deploy/jetson/README.md). The
 * caller writes the preprocessed input into trt_host_in() (pinned memory),
 * calls trt_run(), and reads trt_host_out(). The engine runs on its own CUDA
 * stream at the LOWEST priority: in the brain's process (native/lif_cuda.cu)
 * the brain's kernels are scheduled first; from another process (the camera's,
 * robot/head.py) the GPU time-slices the two. The wait at the end blocks the
 * thread (no spinning CPU core).
 *
 *   make -C native trt   ->  native/libtrt_detect.so
 */
#include <NvInfer.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

using namespace nvinfer1;

namespace {

struct Logger : ILogger {
    void log(Severity s, const char *msg) noexcept override {
        if (s <= Severity::kWARNING) fprintf(stderr, "[trt] %s\n", msg);
    }
} g_log;

struct Det {
    IRuntime *rt = nullptr;
    ICudaEngine *eng = nullptr;
    IExecutionContext *ctx = nullptr;
    cudaStream_t st = nullptr;
    cudaEvent_t done = nullptr;
    Dims in_d{}, out_d{};
    size_t in_n = 0, out_n = 0;
    float *d_in = nullptr, *d_out = nullptr, *h_in = nullptr, *h_out = nullptr;
};

size_t volume(const Dims &d) {
    size_t v = 1;
    for (int i = 0; i < d.nbDims; i++) {
        if (d.d[i] <= 0) return 0;              /* dynamic: not supported */
        v *= (size_t)d.d[i];
    }
    return v;
}

void say(char *err, int n, const char *msg) {
    if (err && n > 0) snprintf(err, (size_t)n, "%s", msg);
}

void destroy(Det *d) {
    if (!d) return;
    delete d->ctx;
    delete d->eng;
    delete d->rt;
    if (d->d_in) cudaFree(d->d_in);
    if (d->d_out) cudaFree(d->d_out);
    if (d->h_in) cudaFreeHost(d->h_in);
    if (d->h_out) cudaFreeHost(d->h_out);
    if (d->done) cudaEventDestroy(d->done);
    if (d->st) cudaStreamDestroy(d->st);
    delete d;
}

}  // namespace

extern "C" {

/* Load a serialized engine. Returns NULL with a message in err on failure. */
void *trt_open(const char *path, char *err, int errlen) try {
    std::ifstream f(path, std::ios::binary);
    if (!f) { say(err, errlen, "cannot read the engine file"); return nullptr; }
    std::vector<char> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    Det *d = new Det();
    d->rt = createInferRuntime(g_log);
    if (!d->rt) { say(err, errlen, "createInferRuntime failed"); destroy(d); return nullptr; }
    d->eng = d->rt->deserializeCudaEngine(buf.data(), buf.size());
    if (!d->eng) { say(err, errlen, "deserializing the engine failed (TensorRT version or GPU mismatch?)"); destroy(d); return nullptr; }
    d->ctx = d->eng->createExecutionContext();
    if (!d->ctx) { say(err, errlen, "createExecutionContext failed"); destroy(d); return nullptr; }
    const char *in_name = nullptr, *out_name = nullptr;
    int nio = d->eng->getNbIOTensors();
    for (int i = 0; i < nio; i++) {
        const char *nm = d->eng->getIOTensorName(i);
        if (d->eng->getTensorDataType(nm) != DataType::kFLOAT) {
            say(err, errlen, "the engine's inputs and outputs must be float32"); destroy(d); return nullptr;
        }
        if (d->eng->getTensorIOMode(nm) == TensorIOMode::kINPUT) {
            if (in_name) { say(err, errlen, "more than one input"); destroy(d); return nullptr; }
            in_name = nm;
        } else {
            if (out_name) { say(err, errlen, "more than one output"); destroy(d); return nullptr; }
            out_name = nm;
        }
    }
    if (!in_name || !out_name) { say(err, errlen, "need one input and one output"); destroy(d); return nullptr; }
    d->in_d = d->eng->getTensorShape(in_name);
    d->out_d = d->eng->getTensorShape(out_name);
    d->in_n = volume(d->in_d);
    d->out_n = volume(d->out_d);
    if (!d->in_n || !d->out_n) { say(err, errlen, "dynamic shapes are not supported"); destroy(d); return nullptr; }
    int least = 0, greatest = 0;
    if (cudaDeviceGetStreamPriorityRange(&least, &greatest) != cudaSuccess ||
        cudaStreamCreateWithPriority(&d->st, cudaStreamNonBlocking, least) != cudaSuccess ||
        cudaEventCreateWithFlags(&d->done, cudaEventBlockingSync | cudaEventDisableTiming) != cudaSuccess ||
        cudaMalloc(&d->d_in, d->in_n * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d->d_out, d->out_n * sizeof(float)) != cudaSuccess ||
        cudaHostAlloc(&d->h_in, d->in_n * sizeof(float), cudaHostAllocDefault) != cudaSuccess ||
        cudaHostAlloc(&d->h_out, d->out_n * sizeof(float), cudaHostAllocDefault) != cudaSuccess) {
        say(err, errlen, cudaGetErrorString(cudaGetLastError()));
        destroy(d);
        return nullptr;
    }
    if (!d->ctx->setTensorAddress(in_name, d->d_in) || !d->ctx->setTensorAddress(out_name, d->d_out)) {
        say(err, errlen, "setTensorAddress failed"); destroy(d); return nullptr;
    }
    return d;
} catch (const std::exception &ex) {          /* never let a C++ exception reach ctypes */
    say(err, errlen, ex.what());
    return nullptr;
}

/* Shape of the input (which = 0) or the output (1); returns the rank. */
int trt_dims(void *h, int which, int64_t *dims, int maxd) {
    const Dims &s = which ? ((Det *)h)->out_d : ((Det *)h)->in_d;
    for (int i = 0; i < s.nbDims && i < maxd; i++) dims[i] = s.d[i];
    return s.nbDims;
}

float *trt_host_in(void *h) { return ((Det *)h)->h_in; }
float *trt_host_out(void *h) { return ((Det *)h)->h_out; }

/* Input (host) -> engine -> output (host). 0 on success, else a CUDA error code
 * (or -1 if TensorRT refused to enqueue). */
int trt_run(void *h) {
    Det *d = (Det *)h;
    cudaError_t e = cudaMemcpyAsync(d->d_in, d->h_in, d->in_n * sizeof(float), cudaMemcpyHostToDevice, d->st);
    if (e != cudaSuccess) return (int)e;
    if (!d->ctx->enqueueV3(d->st)) return -1;
    e = cudaMemcpyAsync(d->h_out, d->d_out, d->out_n * sizeof(float), cudaMemcpyDeviceToHost, d->st);
    if (e == cudaSuccess) e = cudaEventRecord(d->done, d->st);
    if (e == cudaSuccess) e = cudaEventSynchronize(d->done);
    return (int)e;
}

void trt_close(void *h) { destroy((Det *)h); }

}  /* extern "C" */
