/*
 * CUDA whole-brain LIF engine (Shiu et al. 2024): the same C API as
 * native/lif_native.c, so native/lif_native.py loads it unchanged
 * (FLY_NATIVE_LIB=native/liblif_cuda.so). Build: make -C native cuda.
 *
 * Bit-identical to the CPU engine at the default settings:
 *  - every float32 operation per neuron in the same order (-fmad=false: no
 *    fused multiply-adds, as the CPU's -ffp-contract=off);
 *  - synaptic input for a step is summed per target in float64 and cast to
 *    float32 once at delivery (the CPU sums in a float64 hash). Weights are
 *    float32 values within 2^-25 .. 2^17, so float64 atomic sums are exact in
 *    any order (as the CPU's comment explains). With plasticity multipliers
 *    that push weights far below that range, sums could differ in the last
 *    bit from the CPU's order;
 *  - Poisson draws reproduce the CPU's mulberry32 stream: it is counter based,
 *    so draw q of a step is computed directly from the step's starting state;
 *  - spikes per step come out in ascending order (ordered compaction);
 *  - quiescent-block skipping (a CPU optimisation that never changes results)
 *    is not needed: every neuron is integrated every step. The approximate
 *    quiescence tolerance (lif_set_quiesce_tol) is reproduced per 16-neuron
 *    block at the same check steps.
 *
 * Host mirrors: v, g, adapt, gs live on the device; lif_sync_host copies them
 * to host arrays (the Python wrapper calls it when .v/.g/.adapt/.gs are read)
 * and lif_wake_all uploads them back after Python writes. spike_counts is kept
 * on the host from the collected spikes. Plasticity multipliers live in a host
 * array (lif_plastic_enable) and are uploaded by lif_plastic_commit, and in
 * full on the first run after enabling or a reset.
 */
#include <cuda_runtime.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BLK 16
#define PE_SHIFT 14
#define PE_MASK ((1u << PE_SHIFT) - 1)
#define PE_BIAS 8192
#define PE_MAX_N (1 << (32 - PE_SHIFT))
#define QUIESCE_EVERY 32
#define X_EPS 1e-5f
#define TPB 1024                       /* threads per block for neuron kernels */
#define CHUNK 1000                     /* steps per device buffer flush */

#define CK(x) do { cudaError_t _e = (x); if (_e != cudaSuccess) { \
    fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); abort(); } } while (0)

/* per-step counters on the device, so a CUDA graph can replay many steps */
struct DevState { int slot; int npoi; uint32_t rng; int pad; long long step; };

typedef struct lif {
    int n, nnz, D, R, nb, nblk16;
    double dt;
    float ev, eg, kg, v0rest, vth, vrst, poi_w, vfix, qtol, x_eps;
    uint32_t rng;
    long step_count;
    int slot;
    /* device */
    int32_t *d_indptr; uint32_t *d_pe; float *d_wlut, *d_pmult;
    float *d_v, *d_g, *d_adapt, *d_gs, *d_adapt_b, *d_slow_r;
    uint8_t *d_rl, *d_rlen, *d_sil, *d_poi, *d_xf, *d_pinned, *d_flag;
    int32_t *d_poi_q; double *d_poi_p;
    double *d_ring, *d_ring_s;
    int *d_blkcnt, *d_blkoff;
    long long *d_tot;                  /* collected so far (device) */
    int32_t *d_coll; long coll_cap;
    long long *d_sstart; int *d_scount;   /* per step within a chunk */
    int32_t *d_tmp_i; float *d_tmp_f; int tmp_cap;
    /* host */
    float *wlut, *pmult, *h_v, *h_g, *h_adapt, *h_gs;
    int32_t *spike_counts, *spikes, *collected; long ncollected, collected_cap; int nspikes;
    int32_t *poi_idx; double *poi_p; int npoi;
    int ext, plastic, pm_dirty, host_valid, state_dirty;
    float ea, ka, es, ks;
    /* short-term depression of output synapses (lif_set_std; see lif_native.c) */
    float *d_std_f, *d_std_d, *h_std_d; float std_e; int std_on;
    cudaStream_t st;
    DevState *d_st;
    int cfg_version, graph_steps[4], graph_version[4]; cudaGraphExec_t graph[4]; int ngraph;
    /* async driver */
    pthread_t th; int has_th, job_steps;
    pthread_mutex_t mu; pthread_cond_t cv_go, cv_done; int go, done, quit;
    const int16_t *weights_h;
} lif;

/* ------------------------------------------------------------------ kernels */
__device__ __forceinline__ double mulberry_at(uint32_t s0, uint32_t k) {
    uint32_t a = s0 + 0x6D2B79F5u * (k + 1u);
    uint32_t t = (uint32_t)((a ^ (a >> 15)) * (1u | a));
    t = (t + (uint32_t)((t ^ (t >> 7)) * (61u | t))) ^ t;
    return (double)(t ^ (t >> 14)) / 4294967296.0;
}

struct StepArgs {
    int n, ext;
    float ev, eg, kg, v0r, vth, poi_w, vfix, qtol, ka, ks, ea, es, xe;
    const float *std_f; float *std_d; float std_e;     /* std_f NULL = no depression */
};


__global__ void k_advance(DevState *st, int D) {
    st->rng += 0x6D2B79F5u * (uint32_t)st->npoi;
    st->slot = (st->slot + 1) % (D + 1);
    st->step += 1;
}

__global__ void k_update(StepArgs a, const DevState *__restrict__ ds_, float *__restrict__ V, float *__restrict__ G,
                         double *__restrict__ ring, double *__restrict__ ring_s,
                         uint8_t *__restrict__ rl, const uint8_t *__restrict__ is_poi,
                         const int32_t *__restrict__ poi_q, const double *__restrict__ poi_p,
                         float *__restrict__ A, float *__restrict__ GS, uint8_t *__restrict__ xf,
                         const uint8_t *__restrict__ pinned, uint8_t *__restrict__ flag,
                         int *__restrict__ blkcnt) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const bool in = j < a.n;
    const int slot = ds_->slot;
    const int check = (ds_->step % QUIESCE_EVERY) == 0;
    const uint32_t rng0 = ds_->rng;
    float v = 0.0f, g = 0.0f, v2 = 0.0f, g2 = 0.0f, hv = 0.0f, hg = 0.0f;
    int r = 0;
    bool same = true;
    if (in) {
        v = V[j]; g = G[j];
        const size_t o = (size_t)slot * a.n + j;
        const double d = ring[o];
        if (d != 0.0) { g = __fadd_rn(g, (float)d); ring[o] = 0.0; }
        if (a.ext) {
            const double ds = ring_s[o];
            if (ds != 0.0) {
                const float fs = (float)ds;
                if (fs != 0.0f) { GS[j] = __fadd_rn(GS[j], fs); xf[j] = 1; }
                ring_s[o] = 0.0;
            }
        }
        r = rl[j];
        hv = v; hg = g;
        const float tmp = __fmul_rn(g, a.kg);
        float nv = __fmul_rn(v, a.ev);
        nv = __fadd_rn(nv, a.v0r);
        nv = __fadd_rn(nv, tmp);
        const float ng = __fmul_rn(g, a.eg);
        same = (__float_as_uint(nv) == __float_as_uint(v)) && (__float_as_uint(ng) == __float_as_uint(g));
        v2 = nv; g2 = ng;
    }
    /* approximate quiescence (qtol > 0), per 16-neuron block, as the CPU */
    if (check && a.qtol > 0.0f) {
        const int b = j / BLK;
        const bool full = (b + 1) * BLK <= a.n;
        float vmax = in ? v2 : -INFINITY;
        int allsame = same ? 1 : 0;
        int near = in ? ((fabsf(v2 - a.vfix) <= a.qtol) && (fabsf(g2) <= a.qtol) && !r) : 1;
        for (int off = 8; off >= 1; off >>= 1) {
            vmax = fmaxf(vmax, __shfl_xor_sync(0xffffffffu, vmax, off, 16));
            allsame &= __shfl_xor_sync(0xffffffffu, allsame, off, 16);
            near &= __shfl_xor_sync(0xffffffffu, near, off, 16);
        }
        if (in && full && !allsame && vmax < a.vth && !pinned[b] && near) { v2 = a.vfix; g2 = 0.0f; }
    }
    bool spk = false;
    if (in) {
        const bool poi = is_poi[j] != 0;
        bool cand = (v2 > a.vth) && !poi && !r;
        if (a.ext && xf[j]) {
            float aa = A[j], s = GS[j];
            float t2 = __fsub_rn(v2, __fmul_rn(aa, a.ka));
            t2 = __fadd_rn(t2, __fmul_rn(s, a.ks));
            v2 = t2;
            aa = __fmul_rn(aa, a.ea); s = __fmul_rn(s, a.es);
            if (aa < a.xe && s > -a.xe) { aa = 0.0f; s = 0.0f; xf[j] = 0; }
            A[j] = aa; GS[j] = s;
            cand = cand && (v2 > a.vth);
        }
        if (r) { v2 = hv; g2 = hg; rl[j] = (uint8_t)(r - 1); }
        if (poi) {
            const int q = poi_q[j];
            if (mulberry_at(rng0, (uint32_t)q) < poi_p[q]) v2 = __fadd_rn(v2, a.poi_w);
            cand = v2 > a.vth;
        }
        spk = cand;
        V[j] = v2; G[j] = g2;
        if (a.std_f && a.std_f[j] > 0.0f) a.std_d[j] = __fmul_rn(a.std_d[j], a.std_e);
        flag[j] = spk ? 1 : 0;
    }
    const int c = __syncthreads_count(spk);
    if (threadIdx.x == 0) blkcnt[blockIdx.x] = c;
}

/* exclusive scan of block counts; step start/count bookkeeping */
__global__ void k_scan(int nb, const int *__restrict__ blkcnt, int *__restrict__ blkoff,
                       long long *__restrict__ tot, long long *__restrict__ sstart,
                       int *__restrict__ scount, int stepi) {
    __shared__ int sh[1024];
    __shared__ long long base;
    if (threadIdx.x == 0) base = *tot;
    int acc = 0;
    /* serial chunks of 1024 */
    for (int s = 0; s < nb; s += blockDim.x) {
        const int i = s + threadIdx.x;
        const int x = i < nb ? blkcnt[i] : 0;
        sh[threadIdx.x] = x;
        __syncthreads();
        for (int off = 1; off < blockDim.x; off <<= 1) {
            const int y = threadIdx.x >= off ? sh[threadIdx.x - off] : 0;
            __syncthreads();
            sh[threadIdx.x] += y;
            __syncthreads();
        }
        if (i < nb) blkoff[i] = acc + sh[threadIdx.x] - x;
        acc += sh[blockDim.x - 1];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        sstart[stepi] = base;
        scount[stepi] = acc;
        *tot = base + acc;
    }
}

__global__ void k_write(int n, const uint8_t *__restrict__ flag, const int *__restrict__ blkoff,
                        const long long *__restrict__ sstart, int stepi, int32_t *__restrict__ coll) {
    __shared__ int wcnt[TPB / 32];
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const bool f = j < n && flag[j];
    const unsigned m = __ballot_sync(0xffffffffu, f);
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    if (lane == 0) wcnt[w] = __popc(m);
    __syncthreads();
    if (f) {
        int pre = 0;
        for (int k = 0; k < w; k++) pre += wcnt[k];
        pre += __popc(m & ((1u << lane) - 1u));
        coll[sstart[stepi] + blkoff[blockIdx.x] + pre] = j;
    }
}

/* reset spikers (v, g, adaptation, refractory) and scatter their outputs */
__global__ void k_spikes(int n, const DevState *__restrict__ ds_, int D, const int32_t *__restrict__ coll,
                         const long long *__restrict__ sstart,
                         const int *__restrict__ scount, int stepi, int ext, float vrst,
                         float *__restrict__ V, float *__restrict__ G, float *__restrict__ A,
                         const float *__restrict__ adapt_b, uint8_t *__restrict__ xf,
                         uint8_t *__restrict__ rl, const uint8_t *__restrict__ rlen,
                         const uint8_t *__restrict__ sil, const int32_t *__restrict__ indptr,
                         const uint32_t *__restrict__ pe, const float *__restrict__ wlut,
                         const float *__restrict__ pm, const float *__restrict__ slow_r,
                         double *__restrict__ ring, double *__restrict__ ring_s,
                         const float *__restrict__ std_f, float *__restrict__ std_d) {
    const int ns = scount[stepi];
    const long long s0 = sstart[stepi];
    const int out = (ds_->slot + D) % (D + 1);
    const int lane = threadIdx.x & 31;
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int nwarps = (gridDim.x * blockDim.x) >> 5;
    double *rg = ring + (size_t)out * n;
    double *rs = ext ? ring_s + (size_t)out * n : nullptr;
    for (int s = warp; s < ns; s += nwarps) {
        const int i = coll[s0 + s];
        if (lane == 0) {
            V[i] = vrst; G[i] = 0.0f;
            if (ext && adapt_b[i] != 0.0f) { A[i] = __fadd_rn(A[i], adapt_b[i]); xf[i] = 1; }
            rl[i] = rlen[i];
        }
        /* short-term depression: every lane reads the release before lane 0 depletes */
        float rel = 1.0f;
        const bool dep = std_f && std_f[i] > 0.0f;
        if (dep) rel = __fsub_rn(1.0f, std_d[i]);
        __syncwarp();
        if (dep && lane == 0) std_d[i] = __fsub_rn(1.0f, __fmul_rn(rel, std_f[i]));
        if (sil[i]) continue;
        const int a = indptr[i], z = indptr[i + 1];
        for (int q = a + lane; q < z; q += 32) {
            const uint32_t p = pe[q];
            const int tgt = (int)(p >> PE_SHIFT);
            float w = pm ? __fmul_rn(wlut[p & PE_MASK], pm[q]) : wlut[p & PE_MASK];
            if (rel != 1.0f) w = __fmul_rn(w, rel);
            atomicAdd(rg + tgt, (double)w);
            if (ext && w < 0.0f) atomicAdd(rs + tgt, (double)__fmul_rn(w, slow_r[i]));
        }
    }
}

__global__ void k_scatter_f(const int32_t *idx, const float *val, int m, float *dst) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < m) dst[idx[k]] = val[k];
}
__global__ void k_add_g(const int32_t *idx, const float *val, int m, float *G) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < m) atomicAdd(G + idx[k], val[k]);
}
__global__ void k_set_u8(const int32_t *idx, int m, uint8_t *dst, uint8_t val) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < m) dst[idx[k]] = val;
}
__global__ void k_poi_set(const int32_t *idx, int m, uint8_t *is_poi, int32_t *poi_q, uint8_t *rlen,
                          uint8_t *rl, uint8_t *pinned, int on, uint8_t R) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= m) return;
    const int j = idx[k];
    if (on) { is_poi[j] = 1; poi_q[j] = k; rlen[j] = 0; rl[j] = 0; pinned[j / BLK] = 1; }
    else { is_poi[j] = 0; poi_q[j] = -1; rlen[j] = R; }
}

/* ------------------------------------------------------------------ helpers */
static inline int nblocks_for(int m, int t) { return (m + t - 1) / t; }

static void ensure_tmp(lif *e, int m) {
    if (m <= e->tmp_cap) return;
    if (e->d_tmp_i) { cudaFree(e->d_tmp_i); cudaFree(e->d_tmp_f); }
    e->tmp_cap = m * 2 + 1024;
    CK(cudaMalloc(&e->d_tmp_i, (size_t)e->tmp_cap * 4));
    CK(cudaMalloc(&e->d_tmp_f, (size_t)e->tmp_cap * 8));
}

static double drive_coef(double tau_ms, double dt) {
    const double am = 1.0 / 20.0, c = 1.0 / tau_ms;
    if (fabs(am - c) < 1e-12) return am * dt * exp(-am * dt);
    return am * (exp(-c * dt) - exp(-am * dt)) / (am - c);
}

extern "C" {

void lif_reset(lif *e);
long lif_run_collect(lif *e, int steps);

lif *lif_create(int n, int nnz, const int32_t *indptr, const int32_t *indices,
                const int16_t *weights, uint32_t seed, int nthreads) {
    (void)nthreads;
    if (n > PE_MAX_N) return NULL;
    lif *e = (lif *)calloc(1, sizeof(lif));
    e->n = n; e->nnz = nnz;
    const double v_0 = -52.0, t_mbr = 20.0, tau = 5.0, w_syn = 0.275;
    const double dt = getenv("FLY_DT") ? atof(getenv("FLY_DT")) : 0.1;
    e->dt = dt;
    if (fabs(lround(1.8 / dt) * dt - 1.8) > 1e-9 || fabs(lround(2.2 / dt) * dt - 2.2) > 1e-9) { free(e); return NULL; }
    e->D = (int)lround(1.8 / dt); e->R = (int)lround(2.2 / dt);
    uint32_t *pe = (uint32_t *)malloc((size_t)nnz * 4);
    for (int q = 0; q < nnz; q++) {
        if (weights[q] < -PE_BIAS || weights[q] >= PE_BIAS) { free(pe); free(e); return NULL; }
        pe[q] = ((uint32_t)indices[q] << PE_SHIFT) | (uint32_t)(weights[q] + PE_BIAS);
    }
    e->wlut = (float *)malloc((PE_MASK + 1) * sizeof(float));
    for (int c = 0; c <= (int)PE_MASK; c++) e->wlut[c] = (float)((int16_t)(c - PE_BIAS) * w_syn);
    const double a = 1.0 / t_mbr, b = 1.0 / tau;
    const double evd = exp(-a * dt), egd = exp(-b * dt);
    e->ev = (float)evd; e->eg = (float)egd;
    e->kg = (float)(a * (egd - evd) / (a - b));
    e->v0rest = (float)(v_0 * (1.0 - evd));
    e->vth = -45.0f; e->vrst = -52.0f;
    {
        volatile float vf = -52.0f, prev; int it = 0;
        do { prev = vf; float t1 = vf * e->ev; t1 = t1 + e->v0rest; t1 = t1 + 0.0f; vf = t1; } while (vf != prev && ++it < 100000);
        e->vfix = vf;
    }
    e->qtol = 0.0f; e->x_eps = X_EPS;
    e->poi_w = (float)(w_syn * 250.0);
    e->rng = seed;
    e->nb = nblocks_for(n, TPB);
    e->nblk16 = (n + BLK - 1) / BLK;
    /* A BLOCKING stream: the host-side cudaMemcpy/cudaMemset calls below run on
     * the legacy default stream, and a pageable host->device cudaMemcpy may
     * return before its DMA lands. A non-blocking stream would let the next
     * kernel read the old data (seen when several processes share the GPU:
     * wrong silencing / Poisson rates / weights). A blocking stream is ordered
     * after all legacy default-stream work. */
    CK(cudaStreamCreate(&e->st));
    CK(cudaMalloc(&e->d_indptr, (size_t)(n + 1) * 4));
    CK(cudaMemcpy(e->d_indptr, indptr, (size_t)(n + 1) * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&e->d_pe, (size_t)nnz * 4));
    CK(cudaMemcpy(e->d_pe, pe, (size_t)nnz * 4, cudaMemcpyHostToDevice));
    free(pe);
    CK(cudaMalloc(&e->d_wlut, (PE_MASK + 1) * sizeof(float)));
    CK(cudaMemcpy(e->d_wlut, e->wlut, (PE_MASK + 1) * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&e->d_v, (size_t)n * 4)); CK(cudaMalloc(&e->d_g, (size_t)n * 4));
    CK(cudaMalloc(&e->d_rl, n)); CK(cudaMalloc(&e->d_rlen, n)); CK(cudaMalloc(&e->d_sil, n));
    CK(cudaMalloc(&e->d_poi, n)); CK(cudaMalloc(&e->d_flag, n)); CK(cudaMalloc(&e->d_pinned, e->nblk16));
    CK(cudaMalloc(&e->d_poi_q, (size_t)n * 4));
    CK(cudaMalloc(&e->d_poi_p, (size_t)n * 8));
    CK(cudaMalloc(&e->d_ring, (size_t)(e->D + 1) * n * 8));
    CK(cudaMalloc(&e->d_blkcnt, (size_t)e->nb * 4)); CK(cudaMalloc(&e->d_blkoff, (size_t)e->nb * 4));
    CK(cudaMalloc(&e->d_tot, 8));
    CK(cudaMalloc(&e->d_st, sizeof(DevState)));
    e->coll_cap = 4L * 1024 * 1024;
    CK(cudaMalloc(&e->d_coll, (size_t)e->coll_cap * 4));
    CK(cudaMalloc(&e->d_sstart, CHUNK * 8)); CK(cudaMalloc(&e->d_scount, CHUNK * 4));
    CK(cudaMallocHost(&e->h_v, (size_t)n * 4)); CK(cudaMallocHost(&e->h_g, (size_t)n * 4));
    e->spike_counts = (int32_t *)calloc(n, 4);
    e->spikes = (int32_t *)malloc(4096 * 4);
    pthread_mutex_init(&e->mu, NULL); pthread_cond_init(&e->cv_go, NULL); pthread_cond_init(&e->cv_done, NULL);
    lif_reset(e);
    return e;
}

static void *driver_main(void *arg);
static void drop_graphs(lif *e);

void lif_destroy(lif *e) {
    if (e->has_th) {
        pthread_mutex_lock(&e->mu); e->quit = 1; pthread_cond_signal(&e->cv_go); pthread_mutex_unlock(&e->mu);
        pthread_join(e->th, NULL);
    }
    cudaStreamSynchronize(e->st);
    drop_graphs(e);
    void *dp[] = {e->d_st, e->d_indptr, e->d_pe, e->d_wlut, e->d_pmult, e->d_v, e->d_g, e->d_adapt, e->d_gs, e->d_adapt_b,
                  e->d_slow_r, e->d_rl, e->d_rlen, e->d_sil, e->d_poi, e->d_xf, e->d_pinned, e->d_flag, e->d_poi_q,
                  e->d_poi_p, e->d_ring, e->d_ring_s, e->d_blkcnt, e->d_blkoff, e->d_tot, e->d_coll, e->d_sstart,
                  e->d_scount, e->d_tmp_i, e->d_tmp_f, e->d_std_f, e->d_std_d};
    for (size_t k = 0; k < sizeof dp / sizeof dp[0]; k++) if (dp[k]) cudaFree(dp[k]);
    cudaFreeHost(e->h_v); cudaFreeHost(e->h_g);
    if (e->h_adapt) cudaFreeHost(e->h_adapt);
    if (e->h_gs) cudaFreeHost(e->h_gs);
    if (e->h_std_d) cudaFreeHost(e->h_std_d);
    free(e->wlut); free(e->pmult); free(e->spike_counts); free(e->spikes); free(e->collected);
    free(e->poi_idx); free(e->poi_p);
    cudaStreamDestroy(e->st);
    free(e);
}

void lif_reset(lif *e) {
    const int n = e->n;
    for (int i = 0; i < n; i++) { e->h_v[i] = -52.0f; e->h_g[i] = 0.0f; }
    CK(cudaMemcpy(e->d_v, e->h_v, (size_t)n * 4, cudaMemcpyHostToDevice));
    CK(cudaMemset(e->d_g, 0, (size_t)n * 4));
    CK(cudaMemset(e->d_rl, 0, n));
    CK(cudaMemset(e->d_rlen, e->R, n));
    CK(cudaMemset(e->d_sil, 0, n));
    CK(cudaMemset(e->d_poi, 0, n));
    CK(cudaMemset(e->d_pinned, 0, e->nblk16));
    CK(cudaMemset(e->d_poi_q, 0xFF, (size_t)n * 4));
    CK(cudaMemset(e->d_ring, 0, (size_t)(e->D + 1) * n * 8));
    if (e->d_ring_s) CK(cudaMemset(e->d_ring_s, 0, (size_t)(e->D + 1) * n * 8));
    if (e->d_adapt) { CK(cudaMemset(e->d_adapt, 0, (size_t)n * 4)); CK(cudaMemset(e->d_gs, 0, (size_t)n * 4)); CK(cudaMemset(e->d_xf, 0, n)); }
    if (e->d_std_d) CK(cudaMemset(e->d_std_d, 0, (size_t)n * 4));
    if (e->h_adapt) { memset(e->h_adapt, 0, (size_t)n * 4); memset(e->h_gs, 0, (size_t)n * 4); }
    memset(e->spike_counts, 0, (size_t)n * 4);
    e->slot = 0; e->step_count = 0; e->nspikes = 0; e->npoi = 0;
    e->host_valid = 1; e->state_dirty = 1;
    if (e->plastic) e->pm_dirty = 1;
}

void lif_set_seed(lif *e, uint32_t seed) { e->rng = seed; e->state_dirty = 1; }

void lif_set_gain(lif *e, const int16_t *weights, double gain) {
    (void)weights;
    const double w = 0.275 * gain;
    for (int c = 0; c <= (int)PE_MASK; c++) e->wlut[c] = (float)((int16_t)(c - PE_BIAS) * w);
    CK(cudaMemcpy(e->d_wlut, e->wlut, (PE_MASK + 1) * sizeof(float), cudaMemcpyHostToDevice));
}

void lif_set_poisson(lif *e, const int32_t *idx, const double *rates_hz, int m) {
    if (e->npoi) {
        ensure_tmp(e, e->npoi);
        CK(cudaMemcpy(e->d_tmp_i, e->poi_idx, (size_t)e->npoi * 4, cudaMemcpyHostToDevice));
        k_poi_set<<<nblocks_for(e->npoi, 256), 256, 0, e->st>>>(e->d_tmp_i, e->npoi, e->d_poi, e->d_poi_q,
                                                                 e->d_rlen, e->d_rl, e->d_pinned, 0, (uint8_t)e->R);
    }
    CK(cudaMemsetAsync(e->d_pinned, 0, e->nblk16, e->st));
    e->poi_idx = (int32_t *)realloc(e->poi_idx, (size_t)(m ? m : 1) * 4);
    e->poi_p = (double *)realloc(e->poi_p, (size_t)(m ? m : 1) * 8);
    e->npoi = m;
    e->state_dirty = 1;
    for (int q = 0; q < m; q++) {
        double p = rates_hz[q] * e->dt * 1e-3;
        e->poi_p[q] = p < 0 ? 0 : p > 1 ? 1 : p;
        e->poi_idx[q] = idx[q];
    }
    if (m) {
        ensure_tmp(e, m);
        CK(cudaMemcpyAsync(e->d_tmp_i, e->poi_idx, (size_t)m * 4, cudaMemcpyHostToDevice, e->st));
        CK(cudaMemcpyAsync(e->d_poi_p, e->poi_p, (size_t)m * 8, cudaMemcpyHostToDevice, e->st));
        k_poi_set<<<nblocks_for(m, 256), 256, 0, e->st>>>(e->d_tmp_i, m, e->d_poi, e->d_poi_q, e->d_rlen,
                                                           e->d_rl, e->d_pinned, 1, (uint8_t)e->R);
    }
    CK(cudaStreamSynchronize(e->st));
}

int lif_set_poisson_rates(lif *e, const double *rates_hz, int m) {
    if (m != e->npoi) return -1;
    for (int q = 0; q < m; q++) {
        double p = rates_hz[q] * e->dt * 1e-3;
        e->poi_p[q] = p < 0 ? 0 : p > 1 ? 1 : p;
    }
    if (m) CK(cudaMemcpy(e->d_poi_p, e->poi_p, (size_t)m * 8, cudaMemcpyHostToDevice));
    return 0;
}

void lif_silence(lif *e, const int32_t *idx, int m, int on) {
    if (!m) return;
    ensure_tmp(e, m);
    CK(cudaMemcpy(e->d_tmp_i, idx, (size_t)m * 4, cudaMemcpyHostToDevice));
    k_set_u8<<<nblocks_for(m, 256), 256, 0, e->st>>>(e->d_tmp_i, m, e->d_sil, on ? 1 : 0);
    CK(cudaStreamSynchronize(e->st));
}

void lif_set_quiesce_tol(lif *e, float tol_mV) {
    e->cfg_version++;
    e->qtol = tol_mV > 0.0f ? tol_mV : 0.0f;
    e->x_eps = e->qtol > 0.0f ? fmaxf(X_EPS, 10.0f * e->qtol) : X_EPS;
}
void lif_debug_noquiet(int on) { (void)on; }

double lif_dt(lif *e) { return e->dt; }

/* upload the plasticity multipliers (all, or the given positions) */
void lif_plastic_commit(lif *e, const int64_t *idx, int m) {
    if (!e->plastic) return;
    if (m < 0 || !idx) {
        CK(cudaMemcpy(e->d_pmult, e->pmult, (size_t)e->nnz * 4, cudaMemcpyHostToDevice));
        e->pm_dirty = 0;
        return;
    }
    if (!m) return;
    ensure_tmp(e, m);
    int32_t *ti = (int32_t *)malloc((size_t)m * 4); float *tv = (float *)malloc((size_t)m * 4);
    for (int k = 0; k < m; k++) { ti[k] = (int32_t)idx[k]; tv[k] = e->pmult[idx[k]]; }
    CK(cudaMemcpy(e->d_tmp_i, ti, (size_t)m * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(e->d_tmp_f, tv, (size_t)m * 4, cudaMemcpyHostToDevice));
    k_scatter_f<<<nblocks_for(m, 256), 256, 0, e->st>>>(e->d_tmp_i, e->d_tmp_f, m, e->d_pmult);
    CK(cudaStreamSynchronize(e->st));
    free(ti); free(tv);
}

float *lif_plastic_enable(lif *e) {
    if (!e->plastic) {
        e->pmult = (float *)malloc((size_t)e->nnz * 4);
        for (int q = 0; q < e->nnz; q++) e->pmult[q] = 1.0f;
        CK(cudaMalloc(&e->d_pmult, (size_t)e->nnz * 4));
        e->plastic = 1;
        e->cfg_version++;
    }
    e->pm_dirty = 1;               /* Python may write it before the next run */
    return e->pmult;
}

void lif_set_dynamics(lif *e, double tau_adapt_ms, const float *adapt_mV, double tau_slow_ms, const float *slow_ratio) {
    const int n = e->n;
    int on = 0;
    for (int i = 0; i < n && !on; i++) on = adapt_mV[i] > 0 || slow_ratio[i] > 0;
    if (on) {
        if (!e->d_adapt) {
            CK(cudaMalloc(&e->d_adapt, (size_t)n * 4)); CK(cudaMalloc(&e->d_gs, (size_t)n * 4));
            CK(cudaMalloc(&e->d_xf, n)); CK(cudaMalloc(&e->d_adapt_b, (size_t)n * 4));
            CK(cudaMalloc(&e->d_slow_r, (size_t)n * 4));
            CK(cudaMalloc(&e->d_ring_s, (size_t)(e->D + 1) * n * 8));
            CK(cudaMemset(e->d_adapt, 0, (size_t)n * 4)); CK(cudaMemset(e->d_gs, 0, (size_t)n * 4));
            CK(cudaMemset(e->d_xf, 0, n)); CK(cudaMemset(e->d_ring_s, 0, (size_t)(e->D + 1) * n * 8));
            CK(cudaMallocHost(&e->h_adapt, (size_t)n * 4)); CK(cudaMallocHost(&e->h_gs, (size_t)n * 4));
            memset(e->h_adapt, 0, (size_t)n * 4); memset(e->h_gs, 0, (size_t)n * 4);
        }
        CK(cudaMemcpy(e->d_adapt_b, adapt_mV, (size_t)n * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(e->d_slow_r, slow_ratio, (size_t)n * 4, cudaMemcpyHostToDevice));
        e->ea = (float)exp(-e->dt / tau_adapt_ms); e->ka = (float)drive_coef(tau_adapt_ms, e->dt);
        e->es = (float)exp(-e->dt / tau_slow_ms); e->ks = (float)drive_coef(tau_slow_ms, e->dt);
    } else if (e->d_adapt) {
        void *dp[] = {e->d_adapt, e->d_gs, e->d_xf, e->d_adapt_b, e->d_slow_r, e->d_ring_s};
        for (int k = 0; k < 6; k++) cudaFree(dp[k]);
        e->d_adapt = e->d_gs = e->d_adapt_b = e->d_slow_r = NULL; e->d_xf = NULL; e->d_ring_s = NULL;
        cudaFreeHost(e->h_adapt); cudaFreeHost(e->h_gs); e->h_adapt = e->h_gs = NULL;
    }
    e->ext = on;
    e->cfg_version++;           /* pointers and constants captured in graphs changed */
}

void lif_set_std(lif *e, const float *f, double tau_ms) {
    const int n = e->n;
    int on = 0;
    for (int i = 0; i < n && !on; i++) on = f[i] > 0.0f;
    if (on) {
        if (!e->d_std_f) {
            CK(cudaMalloc(&e->d_std_f, (size_t)n * 4)); CK(cudaMalloc(&e->d_std_d, (size_t)n * 4));
            CK(cudaMemset(e->d_std_d, 0, (size_t)n * 4));
            CK(cudaMallocHost(&e->h_std_d, (size_t)n * 4));
        }
        /* re-enabled after being off: start fully recovered, as the CPU engine
         * (which frees its depletion buffer when depression is switched off) */
        if (!e->std_on) CK(cudaMemset(e->d_std_d, 0, (size_t)n * 4));
        CK(cudaMemcpy(e->d_std_f, f, (size_t)n * 4, cudaMemcpyHostToDevice));
        e->std_e = (float)exp(-e->dt / tau_ms);
    }
    e->std_on = on;
    e->cfg_version++;
}
float *lif_std_depletion(lif *e) {
    if (!e->d_std_d) return NULL;
    CK(cudaMemcpy(e->h_std_d, e->d_std_d, (size_t)e->n * 4, cudaMemcpyDeviceToHost));
    return e->h_std_d;
}

float *lif_adapt(lif *e) { return e->h_adapt; }
float *lif_gs(lif *e) { return e->h_gs; }
float *lif_v(lif *e) { return e->h_v; }
float *lif_g(lif *e) { return e->h_g; }
int32_t *lif_spike_counts(lif *e) { return e->spike_counts; }
const int32_t *lif_spikes(lif *e) { return e->spikes; }
const int32_t *lif_collected(lif *e) { return e->collected; }
long lif_step_count(lif *e) { return e->step_count; }
long lif_active_blocks(lif *e) { return e->step_count * e->nblk16; }
int lif_nblocks(lif *e) { return e->nblk16; }
long lif_edges(lif *e) { (void)e; return 0; }
void lif_phase_times(lif *e, double *out3) { (void)e; out3[0] = out3[1] = out3[2] = 0; }
void lif_thread_stats(lif *e, double *out) { (void)e; for (int k = 0; k < 4; k++) out[k] = 0; }

/* device -> host mirrors of v, g, adapt, gs (the wrapper calls this on read) */
void lif_sync_host(lif *e) {
    if (e->host_valid) return;
    const size_t b = (size_t)e->n * 4;
    CK(cudaMemcpyAsync(e->h_v, e->d_v, b, cudaMemcpyDeviceToHost, e->st));
    CK(cudaMemcpyAsync(e->h_g, e->d_g, b, cudaMemcpyDeviceToHost, e->st));
    if (e->d_adapt) {
        CK(cudaMemcpyAsync(e->h_adapt, e->d_adapt, b, cudaMemcpyDeviceToHost, e->st));
        CK(cudaMemcpyAsync(e->h_gs, e->d_gs, b, cudaMemcpyDeviceToHost, e->st));
    }
    CK(cudaStreamSynchronize(e->st));
    e->host_valid = 1;
}

/* after Python wrote to .v / .g (or adapt/gs): upload the host mirrors */
void lif_wake_all(lif *e) {
    if (!e->host_valid) return;       /* nothing was read, so nothing was written */
    const size_t b = (size_t)e->n * 4;
    CK(cudaMemcpy(e->d_v, e->h_v, b, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(e->d_g, e->h_g, b, cudaMemcpyHostToDevice));
    if (e->d_adapt) {
        CK(cudaMemcpy(e->d_adapt, e->h_adapt, b, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(e->d_gs, e->h_gs, b, cudaMemcpyHostToDevice));
    }
}

void lif_add_g(lif *e, const int32_t *idx, const float *vals, int m) {
    if (!m) return;
    ensure_tmp(e, m);
    CK(cudaMemcpy(e->d_tmp_i, idx, (size_t)m * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(e->d_tmp_f, vals, (size_t)m * 4, cudaMemcpyHostToDevice));
    k_add_g<<<nblocks_for(m, 256), 256, 0, e->st>>>(e->d_tmp_i, e->d_tmp_f, m, e->d_g);
    CK(cudaStreamSynchronize(e->st));
    if (e->host_valid) for (int k = 0; k < m; k++) e->h_g[idx[k]] += vals[k];
}

/* enqueue one step; stepi indexes the chunk's bookkeeping arrays. Counters
 * (slot, rng, step) live in DevState on the device, so the same launches can
 * be captured once in a CUDA graph and replayed. */
static void enqueue_step(lif *e, int stepi) {
    StepArgs a;
    a.n = e->n; a.ext = e->ext;
    a.ev = e->ev; a.eg = e->eg; a.kg = e->kg; a.v0r = e->v0rest; a.vth = e->vth; a.poi_w = e->poi_w;
    a.vfix = e->vfix; a.qtol = e->qtol; a.ka = e->ka; a.ks = e->ks; a.ea = e->ea; a.es = e->es; a.xe = e->x_eps;
    a.std_f = e->std_on ? e->d_std_f : NULL; a.std_d = e->d_std_d; a.std_e = e->std_e;
    k_update<<<e->nb, TPB, 0, e->st>>>(a, e->d_st, e->d_v, e->d_g, e->d_ring, e->d_ring_s, e->d_rl, e->d_poi,
                                       e->d_poi_q, e->d_poi_p, e->d_adapt, e->d_gs, e->d_xf, e->d_pinned,
                                       e->d_flag, e->d_blkcnt);
    k_scan<<<1, 1024, 0, e->st>>>(e->nb, e->d_blkcnt, e->d_blkoff, e->d_tot, e->d_sstart, e->d_scount, stepi);
    k_write<<<e->nb, TPB, 0, e->st>>>(e->n, e->d_flag, e->d_blkoff, e->d_sstart, stepi, e->d_coll);
    k_spikes<<<128, 256, 0, e->st>>>(e->n, e->d_st, e->D, e->d_coll, e->d_sstart, e->d_scount, stepi, e->ext,
                                     e->vrst, e->d_v, e->d_g, e->d_adapt, e->d_adapt_b, e->d_xf, e->d_rl,
                                     e->d_rlen, e->d_sil, e->d_indptr, e->d_pe, e->d_wlut,
                                     e->plastic ? e->d_pmult : NULL, e->d_slow_r, e->d_ring, e->d_ring_s,
                                     e->std_on ? e->d_std_f : NULL, e->d_std_d);
    k_advance<<<1, 1, 0, e->st>>>(e->d_st, e->D);
}

/* host mirror of the device counters after `c` steps */
static void advance_host(lif *e, int c) {
    e->rng += 0x6D2B79F5u * (uint32_t)e->npoi * (uint32_t)c;
    e->slot = (e->slot + c) % (e->D + 1);
    e->step_count += c;
}

/* configuration the kernels were captured with (pointers, flags, constants) */
static int config_key(lif *e) {
    return e->cfg_version;
}

static void drop_graphs(lif *e) {
    for (int k = 0; k < e->ngraph; k++) cudaGraphExecDestroy(e->graph[k]);
    e->ngraph = 0;
}

static void launch_steps(lif *e, int c) {
    /* graphs captured under an older configuration are stale */
    if (e->ngraph && e->graph_version[0] != config_key(e)) drop_graphs(e);
    for (int k = 0; k < e->ngraph; k++)
        if (e->graph_steps[k] == c) { CK(cudaGraphLaunch(e->graph[k], e->st)); return; }
    if (c > 100 || e->ngraph == 4) {                  /* long or unusual runs: plain launches */
        for (int k = 0; k < c; k++) enqueue_step(e, k);
        return;
    }
    cudaGraph_t g;
    CK(cudaStreamBeginCapture(e->st, cudaStreamCaptureModeThreadLocal));
    for (int k = 0; k < c; k++) enqueue_step(e, k);
    CK(cudaStreamEndCapture(e->st, &g));
    cudaGraphExec_t ex;
    CK(cudaGraphInstantiate(&ex, g, 0));
    cudaGraphDestroy(g);
    e->graph[e->ngraph] = ex; e->graph_steps[e->ngraph] = c; e->graph_version[e->ngraph] = config_key(e);
    e->ngraph++;
    CK(cudaGraphLaunch(ex, e->st));
}

/* push host counters to the device state (after reset / seed / poisson changes) */
static void push_state(lif *e) {
    DevState h; memset(&h, 0, sizeof h);
    h.slot = e->slot; h.npoi = e->npoi; h.rng = e->rng; h.step = e->step_count;
    CK(cudaMemcpyAsync(e->d_st, &h, sizeof h, cudaMemcpyHostToDevice, e->st));
    CK(cudaStreamSynchronize(e->st));
}

static void collect_host(lif *e, const int32_t *src, long k) {
    if (e->ncollected + k > e->collected_cap) {
        e->collected_cap = (e->ncollected + k) * 2 + 4096;
        e->collected = (int32_t *)realloc(e->collected, (size_t)e->collected_cap * 4);
    }
    memcpy(e->collected + e->ncollected, src, (size_t)k * 4);
    e->ncollected += k;
}

/* run `steps` steps; collect=1 keeps every spike in e->collected */
static long run_steps(lif *e, int steps, int collect) {
    if (e->plastic && e->pm_dirty) {
        CK(cudaMemcpy(e->d_pmult, e->pmult, (size_t)e->nnz * 4, cudaMemcpyHostToDevice));
        e->pm_dirty = 0;
    }
    if (e->state_dirty) { push_state(e); e->state_dirty = 0; }
    e->ncollected = 0;
    long total = 0;
    static __thread int32_t *hbuf = NULL; static __thread long hcap = 0;
    static __thread long long *hstart = NULL; static __thread int *hcount = NULL;
    if (!hstart) { hstart = (long long *)malloc(CHUNK * 8); hcount = (int *)malloc(CHUNK * 4); }
    for (int s = 0; s < steps;) {
        const int c = steps - s < CHUNK ? steps - s : CHUNK;
        CK(cudaMemsetAsync(e->d_tot, 0, 8, e->st));
        launch_steps(e, c);
        advance_host(e, c);
        long long tot = 0;
        CK(cudaMemcpyAsync(&tot, e->d_tot, 8, cudaMemcpyDeviceToHost, e->st));
        CK(cudaStreamSynchronize(e->st));
        if (tot > e->coll_cap) { fprintf(stderr, "lif_cuda: spike buffer overflow (%lld)\n", tot); abort(); }
        if (tot > hcap) { hcap = tot * 2 + 4096; hbuf = (int32_t *)realloc(hbuf, (size_t)hcap * 4); }
        if (tot) CK(cudaMemcpy(hbuf, e->d_coll, (size_t)tot * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hstart, e->d_sstart, (size_t)c * 8, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hcount, e->d_scount, (size_t)c * 4, cudaMemcpyDeviceToHost));
        for (long q = 0; q < tot; q++) e->spike_counts[hbuf[q]]++;
        if (collect) collect_host(e, hbuf, tot);
        /* lif_spikes(): the last step's spikes */
        const int nl = hcount[c - 1];
        e->spikes = (int32_t *)realloc(e->spikes, (size_t)(nl ? nl : 1) * 4);
        memcpy(e->spikes, hbuf + hstart[c - 1], (size_t)nl * 4);
        e->nspikes = nl;
        total += tot;
        s += c;
    }
    e->host_valid = 0;
    return total;
}

int lif_step(lif *e) { run_steps(e, 1, 0); return e->nspikes; }
int lif_run(lif *e, int steps) { return (int)run_steps(e, steps, 0); }
long lif_run_collect(lif *e, int steps) { run_steps(e, steps, 1); return e->ncollected; }

/* async driver: lif_start returns at once; lif_wait blocks (GIL released) */
static void *driver_main(void *arg) {
    lif *e = (lif *)arg;
    pthread_mutex_lock(&e->mu);
    for (;;) {
        while (!e->go && !e->quit) pthread_cond_wait(&e->cv_go, &e->mu);
        if (e->quit) break;
        e->go = 0;
        const int steps = e->job_steps;
        pthread_mutex_unlock(&e->mu);
        lif_run_collect(e, steps);
        pthread_mutex_lock(&e->mu);
        e->done = 1;
        pthread_cond_signal(&e->cv_done);
    }
    pthread_mutex_unlock(&e->mu);
    return NULL;
}

void lif_start(lif *e, int steps) {
    if (!e->has_th) { pthread_create(&e->th, NULL, driver_main, e); e->has_th = 1; }
    pthread_mutex_lock(&e->mu);
    e->job_steps = steps; e->done = 0; e->go = 1;
    pthread_cond_signal(&e->cv_go);
    pthread_mutex_unlock(&e->mu);
}

long lif_wait(lif *e) {
    pthread_mutex_lock(&e->mu);
    while (!e->done) pthread_cond_wait(&e->cv_done, &e->mu);
    pthread_mutex_unlock(&e->mu);
    return e->ncollected;
}

}  /* extern "C" */
