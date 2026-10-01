/*
 * A TensorRT engine runner for the robot's camera models (robot/detector.py:
 * people; robot/faces.py: faces and face embeddings), called through ctypes so
 * the inference never holds Python's GIL.
 *
 * Any number of float32 inputs and outputs. A dynamic input takes its
 * optimisation profile's "opt" shape (fixed for the engine's life). The caller
 * writes each input into trt_host(i) (pinned memory), calls trt_run(), and
 * reads the outputs from trt_host(j). The engine runs on its own CUDA stream
 * at the LOWEST priority: in the brain's process (native/lif_cuda.cu) the
 * brain's kernels are scheduled first; from another process (the camera's,
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

struct Tensor {
    std::string name;
    bool input = false;
    Dims dims{};
    size_t n = 0;
    float *dev = nullptr, *host = nullptr;
};

struct Det {
    IRuntime *rt = nullptr;
    ICudaEngine *eng = nullptr;
    IExecutionContext *ctx = nullptr;
    cudaStream_t st = nullptr;
    cudaEvent_t done = nullptr;
    std::vector<Tensor> io;
};

size_t volume(const Dims &d) {
    size_t v = 1;
    for (int i = 0; i < d.nbDims; i++) {
        if (d.d[i] <= 0) return 0;
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
    for (Tensor &t : d->io) {
        if (t.dev) cudaFree(t.dev);
        if (t.host) cudaFreeHost(t.host);
    }
    if (d->done) cudaEventDestroy(d->done);
    if (d->st) cudaStreamDestroy(d->st);
    delete d;
}

Det *fail(Det *d, char *err, int errlen, const char *msg) {
    say(err, errlen, msg);
    destroy(d);
    return nullptr;
}

}  // namespace

extern "C" {

/* Load a serialized engine. Returns NULL with a message in err on failure. */
void *trt_open(const char *path, char *err, int errlen) {
    Det *d = nullptr;
    try {
        std::ifstream f(path, std::ios::binary);
        if (!f) return fail(d, err, errlen, "cannot read the engine file");
        std::vector<char> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        d = new Det();
        d->rt = createInferRuntime(g_log);
        if (!d->rt) return fail(d, err, errlen, "createInferRuntime failed");
        d->eng = d->rt->deserializeCudaEngine(buf.data(), buf.size());
        if (!d->eng) return fail(d, err, errlen, "deserializing the engine failed (TensorRT version or GPU mismatch?)");
        d->ctx = d->eng->createExecutionContext();
        if (!d->ctx) return fail(d, err, errlen, "createExecutionContext failed");
        const int nio = d->eng->getNbIOTensors();
        d->io.resize(nio);
        for (int i = 0; i < nio; i++) {               /* inputs first: their shapes fix the outputs' */
            Tensor &t = d->io[i];
            t.name = d->eng->getIOTensorName(i);
            const char *nm = t.name.c_str();
            if (d->eng->getTensorDataType(nm) != DataType::kFLOAT)
                return fail(d, err, errlen, "the engine's inputs and outputs must be float32");
            t.input = d->eng->getTensorIOMode(nm) == TensorIOMode::kINPUT;
            if (!t.input) continue;
            Dims s = d->eng->getTensorShape(nm);
            if (!volume(s)) {                         /* dynamic: the profile's opt shape */
                s = d->eng->getProfileShape(nm, 0, OptProfileSelector::kOPT);
                if (!volume(s) || !d->ctx->setInputShape(nm, s))
                    return fail(d, err, errlen, "a dynamic input without a usable optimisation profile");
            }
            t.dims = s;
        }
        for (Tensor &t : d->io) {
            if (!t.input) t.dims = d->ctx->getTensorShape(t.name.c_str());
            t.n = volume(t.dims);
            if (!t.n) return fail(d, err, errlen, "an output's shape is not known");
        }
        int least = 0, greatest = 0;
        if (cudaDeviceGetStreamPriorityRange(&least, &greatest) != cudaSuccess ||
            cudaStreamCreateWithPriority(&d->st, cudaStreamNonBlocking, least) != cudaSuccess ||
            cudaEventCreateWithFlags(&d->done, cudaEventBlockingSync | cudaEventDisableTiming) != cudaSuccess)
            return fail(d, err, errlen, cudaGetErrorString(cudaGetLastError()));
        for (Tensor &t : d->io) {
            if (cudaMalloc(&t.dev, t.n * sizeof(float)) != cudaSuccess ||
                cudaHostAlloc(&t.host, t.n * sizeof(float), cudaHostAllocDefault) != cudaSuccess)
                return fail(d, err, errlen, cudaGetErrorString(cudaGetLastError()));
            memset(t.host, 0, t.n * sizeof(float));
            if (!d->ctx->setTensorAddress(t.name.c_str(), t.dev))
                return fail(d, err, errlen, "setTensorAddress failed");
        }
        return d;
    } catch (const std::exception &ex) {          /* never let a C++ exception reach ctypes */
        return fail(d, err, errlen, ex.what());
    }
}

/* The number of input and output tensors. */
int trt_count(void *h) { return (int)((Det *)h)->io.size(); }

/* Tensor i: its name (into name, namelen bytes), whether it is an input, and
 * its shape (into dims, at most maxd); returns the rank, or -1 for a bad i. */
int trt_info(void *h, int i, char *name, int namelen, int *is_input, int64_t *dims, int maxd) {
    Det *d = (Det *)h;
    if (i < 0 || i >= (int)d->io.size()) return -1;
    const Tensor &t = d->io[i];
    say(name, namelen, t.name.c_str());
    *is_input = t.input;
    for (int k = 0; k < t.dims.nbDims && k < maxd; k++) dims[k] = t.dims.d[k];
    return t.dims.nbDims;
}

/* Tensor i's pinned host buffer (float32, its shape), or NULL. */
float *trt_host(void *h, int i) {
    Det *d = (Det *)h;
    return i >= 0 && i < (int)d->io.size() ? d->io[i].host : nullptr;
}

/* Inputs (host) -> engine -> outputs (host). 0 on success, else a CUDA error
 * code (or -1 if TensorRT refused to enqueue). */
int trt_run(void *h) {
    Det *d = (Det *)h;
    cudaError_t e = cudaSuccess;
    for (const Tensor &t : d->io)
        if (t.input && e == cudaSuccess)
            e = cudaMemcpyAsync(t.dev, t.host, t.n * sizeof(float), cudaMemcpyHostToDevice, d->st);
    if (e != cudaSuccess) return (int)e;
    if (!d->ctx->enqueueV3(d->st)) return -1;
    for (const Tensor &t : d->io)
        if (!t.input && e == cudaSuccess)
            e = cudaMemcpyAsync(t.host, t.dev, t.n * sizeof(float), cudaMemcpyDeviceToHost, d->st);
    if (e == cudaSuccess) e = cudaEventRecord(d->done, d->st);
    if (e == cudaSuccess) e = cudaEventSynchronize(d->done);
    return (int)e;
}

void trt_close(void *h) { destroy((Det *)h); }

}  /* extern "C" */
