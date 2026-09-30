/*
 * CUDA whole-brain LIF engine (Shiu et al. 2024): the same C API as
 * native/lif_native.c, so native/lif_native.py loads it unchanged
 * (FLY_NATIVE_LIB=native/liblif_cuda.so). Build: make -C native cuda.
 *
 * A run of up to 1000 steps is one cooperative kernel launch (k_run, captured
 * once in a CUDA graph per configuration) with one grid barrier per step,
 * written for small GPUs such as the Jetson Orin's, where launches, DRAM
 * round trips and instruction count, not arithmetic, set the speed (see
 * k_run and native/README_CUDA.md).
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
 *  - spikes per step come out in ascending order (the GPU lists them in any
 *    order; the host sorts each step's before anything reads them);
 *  - quiescent-block skipping (a CPU optimisation that never changes results)
 *    is not needed: every neuron is integrated every step. The approximate
 *    quiescence tolerance (lif_set_quiesce_tol) is reproduced per 16-neuron
 *    block at the same check steps.
 *
 * Host mirrors: v, g, adapt, gs live on the device; lif_sync_host copies them
 * to host arrays (the Python wrapper calls it when .v/.g/.adapt/.gs are read)
 * and lif_wake_all uploads them back after Python writes. spike_counts is kept
 * on the host from the collected spikes. The spike buffer, per-step starts and
 * counts are host memory the GPU writes (mapped; the spikes only on an
 * integrated GPU, where host and device share DRAM). Plasticity multipliers live in a host
 * array (lif_plastic_enable) and are uploaded by lif_plastic_commit, and in
 * full on the first run after enabling or a reset.
 */
#include <cooperative_groups.h>
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
/* per-neuron flag byte (k_run): refractory steps left, Poisson-driven, has
 * adaptation / slow inhibition state */
#define FL_RL 63u
#define FL_POI 64u
#define FL_XF 128u
#define RB 512                         /* threads per block of the run kernel */
#define WPB (RB / 32)
#define SCAT_U 4                       /* synapses per lane in flight when scattering */
#define SCAT_G 16                      /* at most this many warps per spike */
#define CHUNK 1000                     /* steps per device buffer flush */
#define NGRAPH 8                       /* cached launch graphs (launch_steps) */

namespace cg = cooperative_groups;

/* A CUDA error while the engine runs leaves its device state unusable (and a
 * sticky error, e.g. an illegal address, poisons the whole context), so it
 * stops the process with a message. Creating an engine is different: running
 * out of GPU memory there is ordinary (several processes share one GPU), so
 * lif_create uses CKC and returns NULL, which Python turns into an exception. */
#define CK(x) do { cudaError_t _e = (x); if (_e != cudaSuccess) { \
    fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); abort(); } } while (0)
#define CKC(x) do { cudaError_t _e = (x); if (_e != cudaSuccess) { \
    fprintf(stderr, "lif_cuda: engine not created: CUDA %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); \
    goto fail; } } while (0)

/* per-step counters on the device, carried from one run to the next */
struct DevState { int slot; int npoi; uint32_t rng; int pad; long long step; };

typedef struct lif {
    int n, npad, nnz, D, R, nblk16;      /* npad: n rounded up to whole 32-neuron words */
    int nbk, nwords, lwb, smem;        /* k_run: blocks, 32-neuron words, words per block, shared bytes (0: off) */
    int watchdog;                      /* the GPU has a kernel time limit (drives a display) */
    double dt;
    float ev, eg, kg, v0rest, vth, vrst, poi_w, vfix, qtol, x_eps;
    uint32_t rng;
    long step_count;
    int slot;
    /* device */
    int32_t *d_indptr; uint32_t *d_pe; float *d_wlut, *d_pmult;
    float *d_state;                    /* v, g, adapt, gs: one block, kept in L2 where it fits */
    float *d_v, *d_g, *d_adapt, *d_gs, *d_adapt_b, *d_slow_r;
    uint8_t *d_fl, *d_rlen, *d_sil, *d_pinned;
    int2 *d_poi_n;                     /* per neuron: Poisson draw index and threshold (poi_entry) */
    int32_t *d_poi_idx; int2 *h_poi_n; int poi_cap;   /* the Poisson set on the device, its entries */
    double *d_ring, *d_ring_s;
    uint32_t *d_touched;               /* per ring slot: bit per neuron with input pending */
    int *d_cnt; float *d_reln; int32_t *d_sl;   /* k_run: per-step counts, releases, device spike list */
    /* per chunk: spike total, each step's start and count, the spikes. Host
     * memory the GPU writes (mapped), so no copies back; the spikes too on an
     * integrated GPU (Jetson), where host and device share the same DRAM. */
    long long *h_tot, *d_tot;
    int32_t *d_coll, *h_coll; long coll_cap; int integrated;
    long long *h_sstart, *d_sstart; int *h_scount, *d_scount;
    int32_t *d_tmp_i; float *d_tmp_f; int tmp_cap;
    /* host */
    float *wlut, *pmult, *h_v, *h_g, *h_adapt, *h_gs;
    int32_t *spike_counts, *spikes, *collected; long ncollected, collected_cap; int nspikes;
    int32_t *hbuf; long hcap;          /* one chunk's spikes, copied back */
    uint32_t *sortbits;                /* sort_step */
    int32_t *poi_idx; double *poi_p; int npoi;
    int ext, plastic, pm_dirty, host_valid, state_dirty;
    float ea, ka, es, ks;
    /* short-term depression of output synapses (lif_set_std; see lif_native.c) */
    float *d_std_f, *d_std_d, *h_std_d; long long *d_std_t, *h_std_t; double std_base; int std_on;
    cudaStream_t st;
    struct RunArgs *graph_args; cudaGraphExec_t graph[NGRAPH]; int ngraph, nograph;   /* launch_steps */
    DevState *d_st;
    /* async driver */
    pthread_t th; int has_th, job_steps;
    pthread_mutex_t mu; pthread_cond_t cv_go, cv_done; int go, done, quit;
    const int16_t *weights_h;
} lif;

/* ------------------------------------------------------------------ kernels */
/* draw k of a step as the 32-bit integer u; the CPU's uniform is u / 2^32 */
__device__ __forceinline__ uint32_t mulberry_u(uint32_t s0, uint32_t k) {
    uint32_t a = s0 + 0x6D2B79F5u * (k + 1u);
    uint32_t t = (uint32_t)((a ^ (a >> 15)) * (1u | a));
    t = (t + (uint32_t)((t ^ (t >> 7)) * (61u | t))) ^ t;
    return t ^ (t >> 14);
}

/* A Poisson neuron's draw q and threshold: the CPU fires when u / 2^32 < p,
 * i.e. (exactly: u is an integer and p * 2^32 is exact in double) when
 * u < ceil(p * 2^32), i.e. u <= that - 1. q = -1 when p = 0 (never fires). */
static int2 poi_entry(int q, double p) {
    const double x = ceil(p * 4294967296.0);
    if (!(x > 0.0)) return make_int2(-1, 0);          /* p = 0, or NaN (never fires on the CPU) */
    return make_int2(q, (int)(uint32_t)(x - 1.0));
}

struct StepArgs {
    int n, ext;
    float ev, eg, kg, v0r, vth, poi_w, vfix, qtol, ka, ks, ea, es, xe;
};

/* base^gap by binary powering, IEEE multiplies only: bit-identical with the
 * CPU engine's std_decay (lif_native.c) */
__host__ __device__ static inline double std_decay_d(double base, long long gap) {
    double r = 1.0;
    while (gap) {
#ifdef __CUDA_ARCH__
        if (gap & 1) r = __dmul_rn(r, base); base = __dmul_rn(base, base);
#else
        if (gap & 1) r *= base; base *= base;
#endif
        gap >>= 1;
    }
    return r;
}

/* Everything one run of `c` steps needs (k_run). */
struct RunArgs {
    StepArgs a;
    int D, c, nwords, lwb;             /* lwb: words per block (SH: the shared arrays' size) */
    long long cap;
    float vrst; double std_base;
    DevState *st;
    float *V, *G, *A, *GS; const float *adapt_b, *slow_r;
    uint8_t *fl; const uint8_t *rlen, *sil, *pinned;
    const int2 *poi_n;
    double *ring, *ring_s; uint32_t *touched;
    const int32_t *indptr; const uint32_t *pe; const float *wlut, *pm;
    const float *std_f; float *std_d; long long *std_t;
    int *cnt; float *reln;             /* per step: spike count; per neuron: release (2 x npad) */
    int npad;
    int32_t *sl;                       /* the chunk's spikes (device; = coll on a discrete GPU) */
    long long *tot; int32_t *coll; long long *sstart; int *scount;
};

/*
 * One cooperative launch runs `c` steps (the CPU engine's step, reordered for
 * a small GPU, where every pass over the neurons' state costs DRAM bandwidth
 * and instructions). Each 32-neuron word belongs to one warp (words are dealt
 * round-robin, which balances the load), so a neuron's state is only ever
 * touched by one warp. Per step:
 *
 *  U  each warp integrates its neurons; a spiking neuron is reset at once
 *     (v, g, adaptation, refractory period, depression: what the CPU does
 *     after the update) and appended to the step's spike list. The delay ring
 *     is read only where the slot's `touched` bit is set.
 *  --- grid barrier ---
 *  S  the step's spikes are scattered into the delay ring (setting the
 *     target's bit in that slot's `touched` words).
 *
 * S(k) and U(k+1) need no barrier between them: S(k) writes ring slot
 * slot_k + D, which U reads D steps later; the per-step spike counts are
 * separate and the releases double-buffered. So one grid barrier per step.
 * A step's spikes are listed in no particular order; the host sorts each
 * step's (few) spikes, which gives the CPU engine's ascending order.
 */
template <bool EXT, bool QT, bool SH>
__global__ void __launch_bounds__(RB) k_run(RunArgs ra) {
    cg::grid_group grid = cg::this_grid();
    const StepArgs &a = ra.a;
    const int n = a.n, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int nw = gridDim.x * WPB;                    /* warps in the grid */
    const int wstep = nw;                              /* a warp's words: w, w + nw, ... */
    const int w0 = blockIdx.x + warp * gridDim.x;      /* word i of block b is b + i * nbk */
    /* SH: the block's v and flags live in shared memory for the whole launch
     * (loaded here, written back at the end); each warp only ever touches
     * its own words, so no block barrier is needed */
    extern __shared__ float shv[];
    uint8_t *shf = (uint8_t *)(shv + (size_t)ra.lwb * 32);
    if (SH)
        for (int w = w0, i = warp; w < ra.nwords; w += wstep, i += WPB) {
            shv[i * 32 + lane] = ra.V[w * 32 + lane]; shf[i * 32 + lane] = ra.fl[w * 32 + lane];
        }
    int slot = ra.st->slot;
    uint32_t rng = ra.st->rng;
    long long stp = ra.st->step;
    const int npoi = ra.st->npoi;
    long long base = 0;                /* each launch is one chunk */
    for (int k = 0; k < ra.c; k++) {
        const int par = k & 1;
        /* ---------------- U: update ---------------- */
        uint32_t *tbase = ra.touched + slot * ra.nwords;
        float *reln = ra.reln + par * ra.npad;
        /* software-pipelined: the next word's touched bits, v, g and flags
         * are loaded while this word is computed (a word is otherwise two
         * dependent DRAM round trips) */
        uint32_t tw_n = 0, fl_n = 0;
        float v_n = 0.0f, g_n = 0.0f, a_n = 0.0f, s_n = 0.0f;
        if (w0 < ra.nwords) {
            const int jn = w0 * 32 + lane;
            tw_n = tbase[w0]; g_n = ra.G[jn];
            if (!SH) { v_n = ra.V[jn]; fl_n = ra.fl[jn]; }
            if (EXT) { a_n = ra.A[jn]; s_n = ra.GS[jn]; }
        }
        for (int w = w0, i = warp; w < ra.nwords; w += wstep, i += WPB) {
            uint32_t *twp = tbase + w;
            /* the arrays are padded to whole words: padding neurons rest at
             * -52 mV with no inputs and never spike */
            const int j = w * 32 + lane, sj = i * 32 + lane;
            const uint32_t tw = tw_n, fl = SH ? shf[sj] : fl_n;
            const float v = SH ? shv[sj] : v_n;
            float g = g_n;
            /* adaptation state: every neuron's, one word ahead (at rest ~1 in
             * 15 neurons has some, and a warp waiting for it costs more
             * than the extra reads) */
            const float a0 = a_n, gs0 = s_n;
            if (w + wstep < ra.nwords) {
                const int jn = (w + wstep) * 32 + lane;
                tw_n = tbase[w + wstep]; g_n = ra.G[jn];
                if (!SH) { v_n = ra.V[jn]; fl_n = ra.fl[jn]; }
                if (EXT) { a_n = ra.A[jn]; s_n = ra.GS[jn]; }
            }
            const int r = fl & FL_RL;
            const bool poi = fl & FL_POI;
            const bool xf0 = EXT && (fl & FL_XF);
            /* the loads that depend on them, only where needed: at rest
             * a step's traffic otherwise overflows L2 on a small GPU */
            const bool tch = (tw >> lane) & 1u;
            const int o = slot * n + j;
            double d = 0.0, ds = 0.0;
            int2 pn = make_int2(-1, 0);
            if (tch) { d = ra.ring[o]; if (EXT) ds = ra.ring_s[o]; }
            if (poi) pn = ra.poi_n[j];
            bool xfv = xf0, gsl = false;
            float gs = 0.0f;
            if (tch) {
                if (d != 0.0) { g = __fadd_rn(g, (float)d); ra.ring[o] = 0.0; }
                if (EXT && ds != 0.0) {
                    const float fs = (float)ds;
                    if (fs != 0.0f) { gs = __fadd_rn(gs0, fs); gsl = true; xfv = true; }
                    ra.ring_s[o] = 0.0;
                }
            }
            if (tw) { __syncwarp(); if (lane == 0) *twp = 0u; }
            /* the CPU engine's float32 operations, in its order */
            const float hv = v, hg = g;
            const float tmp = __fmul_rn(g, a.kg);
            float v2 = __fadd_rn(__fadd_rn(__fmul_rn(v, a.ev), a.v0r), tmp);
            float g2 = __fmul_rn(g, a.eg);
            if (QT && (stp % QUIESCE_EVERY) == 0) {
                /* approximate quiescence, per 16-neuron block, as the CPU */
                const bool in = j < n;
                const bool same = (__float_as_uint(v2) == __float_as_uint(v)) && (__float_as_uint(g2) == __float_as_uint(g));
                const int b = j / BLK;
                const bool full = (b + 1) * BLK <= n;
                float vmax = in ? v2 : -INFINITY;
                int allsame = same ? 1 : 0;
                int near = in ? ((fabsf(v2 - a.vfix) <= a.qtol) && (fabsf(g2) <= a.qtol) && !r) : 1;
                for (int off = 8; off >= 1; off >>= 1) {
                    vmax = fmaxf(vmax, __shfl_xor_sync(0xffffffffu, vmax, off, 16));
                    allsame &= __shfl_xor_sync(0xffffffffu, allsame, off, 16);
                    near &= __shfl_xor_sync(0xffffffffu, near, off, 16);
                }
                if (in && full && !allsame && vmax < a.vth && !ra.pinned[b] && near) { v2 = a.vfix; g2 = 0.0f; }
            }
            bool cand = (v2 > a.vth) && !poi && !r;
            bool xran = false;
            float aa = 0.0f;
            if (EXT && xfv) {
                float s2 = gsl ? gs : gs0;
                aa = a0;
                v2 = __fadd_rn(__fsub_rn(v2, __fmul_rn(aa, a.ka)), __fmul_rn(s2, a.ks));
                aa = __fmul_rn(aa, a.ea); s2 = __fmul_rn(s2, a.es);
                if (aa < a.xe && s2 > -a.xe) { aa = 0.0f; s2 = 0.0f; xfv = false; }
                ra.GS[j] = s2;
                xran = true;
                cand = cand && (v2 > a.vth);
            }
            int rn = r;
            if (r) { v2 = hv; g2 = hg; rn = r - 1; }
            if (poi) {
                if (pn.x >= 0 && mulberry_u(rng, (uint32_t)pn.x) <= (uint32_t)pn.y) v2 = __fadd_rn(v2, a.poi_w);
                cand = v2 > a.vth;
            }
            const bool spk = cand;
            float rel = 1.0f;
            if (spk) {
                /* the reset the CPU engine applies to each spiker after the update */
                v2 = ra.vrst; g2 = 0.0f; rn = ra.rlen[j];
                if (EXT && ra.adapt_b[j] != 0.0f) {
                    aa = __fadd_rn(xran ? aa : a0, ra.adapt_b[j]);
                    xran = true; xfv = true;
                }
                /* short-term depression, decayed lazily as in the CPU engine
                 * (brought up to date when the neuron spikes); the release
                 * this spike uses travels with it to the scatter */
                if (ra.std_f) {
                    const float f = ra.std_f[j];
                    if (f > 0.0f) {
                        float dd = ra.std_d[j];
                        const long long gap = stp - ra.std_t[j];
                        if (gap > 0 && dd != 0.0f) dd = (float)__dmul_rn((double)dd, std_decay_d(ra.std_base, gap));
                        rel = __fsub_rn(1.0f, dd);
                        ra.std_d[j] = __fsub_rn(1.0f, __fmul_rn(rel, f)); ra.std_t[j] = stp;
                    }
                }
            }
            if (EXT && xran) ra.A[j] = aa;
            const uint32_t fl2 = (uint32_t)rn | (fl & FL_POI) | (EXT ? (xfv ? FL_XF : 0u) : (fl & FL_XF));
            if (SH) { if (fl2 != fl) shf[sj] = (uint8_t)fl2; shv[sj] = v2; }
            else { if (fl2 != fl) ra.fl[j] = (uint8_t)fl2; ra.V[j] = v2; }
            ra.G[j] = g2;
            const unsigned m = __ballot_sync(0xffffffffu, spk);
            if (m) {
                int at = 0;
                if (lane == 0) at = atomicAdd(ra.cnt + k, __popc(m));
                at = __shfl_sync(0xffffffffu, at, 0);
                if (spk) {
                    const long long q = base + at + __popc(m & ((1u << lane) - 1u));
                    if (q < ra.cap) { ra.sl[q] = j; if (ra.coll != ra.sl) ra.coll[q] = j; }   /* cap: see lif_create */
                    if (ra.std_f) reln[j] = rel;
                }
            }
        }
        grid.sync();
        /* ---------------- S: scatter ---------------- */
        const int total = __ldcg(ra.cnt + k);
        const long long room = ra.cap - base > 0 ? ra.cap - base : 0;   /* never read past the buffer */
        const int ns = (long long)total < room ? total : (int)room;
        if (blockIdx.x == 0 && threadIdx.x == 0) { ra.sstart[k] = base; ra.scount[k] = total; }
        const int out = (slot + ra.D) % (ra.D + 1);
        double *rg = ra.ring + (size_t)out * n;
        double *rs = EXT ? ra.ring_s + (size_t)out * n : nullptr;
        uint32_t *tch = ra.touched + (size_t)out * ra.nwords;
        /* each spike's synapses are split over G warps (all warps share few
         * spikes; a storm gets one warp per spike), spread over the SMs */
        const int G = ns ? max(1, min(SCAT_G, nw / ns)) : 1;
        for (int t = warp * gridDim.x + blockIdx.x; t < ns * G; t += nw) {
            const int i = __ldcg(ra.sl + base + t / G), part = t % G;
            const float rel = ra.std_f ? __ldcg(reln + i) : 1.0f;
            if (ra.sil[i]) continue;
            const int qa = ra.indptr[i], qz = ra.indptr[i + 1];
            const float sr = EXT ? ra.slow_r[i] : 0.0f;
            const int step = 32 * G;
            for (int q0 = qa + part * 32 + lane; q0 < qz; q0 += step * SCAT_U) {
                uint32_t p[SCAT_U]; float w[SCAT_U];
#pragma unroll
                for (int u = 0; u < SCAT_U; u++) {        /* loads first: several in flight */
                    const int q = q0 + step * u;
                    if (q < qz) { p[u] = ra.pe[q]; w[u] = ra.pm ? ra.pm[q] : 1.0f; }
                }
#pragma unroll
                for (int u = 0; u < SCAT_U; u++) {
                    if (q0 + step * u >= qz) break;
                    const int tgt = (int)(p[u] >> PE_SHIFT);
                    float x = ra.pm ? __fmul_rn(ra.wlut[p[u] & PE_MASK], w[u]) : ra.wlut[p[u] & PE_MASK];
                    if (rel != 1.0f) x = __fmul_rn(x, rel);
                    atomicAdd(rg + tgt, (double)x);
                    if (EXT && x < 0.0f) atomicAdd(rs + tgt, (double)__fmul_rn(x, sr));
                    atomicOr(tch + (tgt >> 5), 1u << (tgt & 31));
                }
            }
        }
        base += total;
        rng += 0x6D2B79F5u * (uint32_t)npoi;
        slot = (slot + 1) % (ra.D + 1);
        stp += 1;
    }
    if (SH)
        for (int w = w0, i = warp; w < ra.nwords; w += wstep, i += WPB) {
            ra.V[w * 32 + lane] = shv[i * 32 + lane]; ra.fl[w * 32 + lane] = shf[i * 32 + lane];
        }
    /* every block read the counters before the first barrier */
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        ra.st->slot = slot; ra.st->rng = rng; ra.st->step = stp;
        *ra.tot = base;
    }
}

__global__ void k_scatter_f(const int32_t *idx, const float *val, int m, float *dst) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < m) dst[idx[k]] = val[k];
}
__global__ void k_add_g_seq(const int32_t *idx, const float *val, int m, float *G) {
    for (int k = 0; k < m; k++) G[idx[k]] = __fadd_rn(G[idx[k]], val[k]);
}
__global__ void k_add_g(const int32_t *idx, const float *val, int m, float *G) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < m) atomicAdd(G + idx[k], val[k]);
}
__global__ void k_set_u8(const int32_t *idx, int m, uint8_t *dst, uint8_t val) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < m) dst[idx[k]] = val;
}
__global__ void k_poi_n(const int32_t *idx, const int2 *val, int m, int2 *dst) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < m) dst[idx[k]] = val[k];
}
__global__ void k_fl_and(uint8_t *fl, int m, uint8_t mask) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < m) fl[k] &= mask;
}
__global__ void k_poi_set(const int32_t *idx, int m, uint8_t *fl, uint8_t *rlen,
                          uint8_t *pinned, int on, uint8_t R) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= m) return;
    const int j = idx[k];
    /* on: Poisson-driven, no refractory period (and none left) */
    if (on) { fl[j] = (uint8_t)((fl[j] & FL_XF) | FL_POI); rlen[j] = 0; pinned[j / BLK] = 1; }
    else { fl[j] = (uint8_t)(fl[j] & ~FL_POI); rlen[j] = R; }
}

/* ------------------------------------------------------------------ helpers */
static inline int nblocks_for(int m, int t) { return (m + t - 1) / t; }

static const void *run_kernel(int ext, int qt, int sh) {
    if (sh) return ext ? (qt ? (const void *)k_run<true, true, true> : (const void *)k_run<true, false, true>)
                       : (qt ? (const void *)k_run<false, true, true> : (const void *)k_run<false, false, true>);
    return ext ? (qt ? (const void *)k_run<true, true, false> : (const void *)k_run<true, false, false>)
               : (qt ? (const void *)k_run<false, true, false> : (const void *)k_run<false, false, false>);
}
/* blocks per SM every variant of k_run (with or without shared state) allows
 * with `smem` dynamic shared bytes; 0 on error */
static int occupancy(int smem, int sh, int maxsm) {
    int per = 1 << 30;
    for (int k = 0; k < 4; k++) {
        const void *f = run_kernel(k & 1, k >> 1, sh);
        /* the limit is per kernel for the whole process: always the device's
         * maximum, so one engine never lowers it under another */
        if (smem > 48 * 1024 && cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, maxsm) != cudaSuccess) {
            cudaGetLastError(); return 0;
        }
        int p = 0;
        if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&p, f, RB, smem) != cudaSuccess) { cudaGetLastError(); return 0; }
        if (p < per) per = p;
    }
    return per;
}

static int cmp_i32(const void *a, const void *b) {
    const int32_t x = *(const int32_t *)a, y = *(const int32_t *)b;
    return (x > y) - (x < y);
}

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
void lif_destroy(lif *e);
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
    if (e->D < 2) { free(e); return NULL; }         /* k_run overlaps a step's scatter with the next update */
    if (e->R > (int)FL_RL) { fprintf(stderr, "lif_cuda: FLY_DT too small (refractory period > %u steps)\n", FL_RL); free(e); return NULL; }
    e->npad = (n + 31) / 32 * 32;
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
    e->nblk16 = (n + BLK - 1) / BLK;
    /* A BLOCKING stream: the host-side cudaMemcpy/cudaMemset calls below run on
     * the legacy default stream, and a pageable host->device cudaMemcpy may
     * return before its DMA lands. A non-blocking stream would let the next
     * kernel read the old data (seen when several processes share the GPU:
     * wrong silencing / Poisson rates / weights). A blocking stream is ordered
     * after all legacy default-stream work. */
    CKC(cudaStreamCreate(&e->st));
    CKC(cudaMalloc(&e->d_indptr, (size_t)(n + 1) * 4));
    CKC(cudaMemcpy(e->d_indptr, indptr, (size_t)(n + 1) * 4, cudaMemcpyHostToDevice));
    CKC(cudaMalloc(&e->d_pe, (size_t)nnz * 4));
    CKC(cudaMemcpy(e->d_pe, pe, (size_t)nnz * 4, cudaMemcpyHostToDevice));
    free(pe); pe = NULL;
    CKC(cudaMalloc(&e->d_wlut, (PE_MASK + 1) * sizeof(float)));
    CKC(cudaMemcpy(e->d_wlut, e->wlut, (PE_MASK + 1) * sizeof(float), cudaMemcpyHostToDevice));
    CKC(cudaMalloc(&e->d_state, (size_t)e->npad * 16));
    e->d_v = e->d_state; e->d_g = e->d_state + e->npad;    /* adapt, gs: + 2 npad, + 3 npad (lif_set_dynamics) */
    CKC(cudaMalloc(&e->d_fl, e->npad)); CKC(cudaMalloc(&e->d_rlen, n)); CKC(cudaMalloc(&e->d_sil, n));
    CKC(cudaMalloc(&e->d_pinned, e->nblk16));
    CKC(cudaMalloc(&e->d_poi_n, (size_t)n * 8));
    CKC(cudaMalloc(&e->d_ring, (size_t)(e->D + 1) * n * 8));
    {
        /* k_run: as many blocks as can be resident at once (a cooperative
         * launch needs that), each warp owning wpw consecutive 32-neuron words */
        int dev, sms, per;
        CKC(cudaGetDevice(&dev));
        CKC(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
        int coop = 0;
        CKC(cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, dev));
        if (!coop) { fprintf(stderr, "lif_cuda: the GPU does not support cooperative launches\n"); goto fail; }
        CKC(cudaDeviceGetAttribute(&e->watchdog, cudaDevAttrKernelExecTimeout, dev));
        per = occupancy(0, 0, 0);                           /* blocks per SM, registers allow */
        const char *bp = getenv("FLY_CUDA_BLOCKS_PER_SM");     /* tuning */
        if (bp && atoi(bp) > 0 && atoi(bp) < per) per = atoi(bp);
        CKC(cudaDeviceGetAttribute(&e->integrated, cudaDevAttrIntegrated, dev));
        e->nwords = (n + 31) / 32;
        /* keep v, g, adapt, gs resident in L2 (persisting accesses): on a
         * small GPU every step would otherwise stream them from DRAM */
        int maxp = 0, maxw = 0;
        cudaDeviceGetAttribute(&maxp, cudaDevAttrMaxPersistingL2CacheSize, dev);
        cudaDeviceGetAttribute(&maxw, cudaDevAttrMaxAccessPolicyWindowSize, dev);
        if (maxp > 0 && maxw > 0 && !getenv("FLY_CUDA_NO_L2PIN")) {
            const char *pb = getenv("FLY_CUDA_L2PIN_FLOATS");            /* tuning: floats per neuron pinned */
            const size_t want = (size_t)e->npad * 4 * (pb ? atoi(pb) : 4);
            const size_t win = want < (size_t)maxw ? want : (size_t)maxw;
            const size_t lim = win < (size_t)maxp ? win : (size_t)maxp;
            if (cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, lim) == cudaSuccess) {
                cudaStreamAttrValue at;
                memset(&at, 0, sizeof at);
                at.accessPolicyWindow.base_ptr = e->d_state;
                at.accessPolicyWindow.num_bytes = win;
                at.accessPolicyWindow.hitRatio = (float)lim / (float)win;
                at.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
                at.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
                cudaStreamSetAttribute(e->st, cudaStreamAttributeAccessPolicyWindow, &at);
            }
            cudaGetLastError();                               /* optional: never fatal */
        }
        e->nbk = per * sms;
        if (e->nbk < 1) { fprintf(stderr, "lif_cuda: k_run does not fit on the GPU\n"); goto fail; }
        /* keep each block's v and flags in shared memory (5 bytes a neuron)
         * when they fit with as many blocks per SM as the registers allow */
        int maxsm = 0;
        CKC(cudaDeviceGetAttribute(&maxsm, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
        e->lwb = (e->nwords + e->nbk - 1) / e->nbk;
        e->smem = e->lwb * 32 * 5;
        if (getenv("FLY_CUDA_NO_SMEM") || e->smem > maxsm || occupancy(e->smem, 1, maxsm) < per) e->smem = 0;
    }
    CKC(cudaMalloc(&e->d_touched, (size_t)(e->D + 1) * e->nwords * 4));
    CKC(cudaMalloc(&e->d_cnt, CHUNK * 4));
    CKC(cudaMalloc(&e->d_reln, (size_t)2 * e->npad * 4));
    CKC(cudaHostAlloc(&e->h_tot, 8, cudaHostAllocMapped));
    CKC(cudaHostGetDevicePointer(&e->d_tot, e->h_tot, 0));
    CKC(cudaMalloc(&e->d_st, sizeof(DevState)));
    /* Spike buffer. After a spike a neuron is refractory for R steps, so it
     * spikes at most c / R + 1 times in c steps -- except Poisson-driven
     * neurons, which have no refractory period (rlen = 0) and can spike every
     * step. run_steps() therefore shortens a chunk when the Poisson set is
     * large, so that npoi * c + (n - npoi) * (c / R + 1) <= coll_cap always
     * holds (chunk_steps). (A fixed 4M-spike buffer overflowed in a whole-
     * brain storm at FLY_DT 0.2; the review of 2026-09-26 found that the first
     * fix, sized for refractory neurons only, could still overflow.) */
    e->coll_cap = (long)n * (CHUNK / e->R + 1);
    CKC(cudaMalloc(&e->d_sl, (size_t)e->coll_cap * 4));
    if (e->integrated) {
        CKC(cudaHostAlloc(&e->h_coll, (size_t)e->coll_cap * 4, cudaHostAllocMapped));
        CKC(cudaHostGetDevicePointer(&e->d_coll, e->h_coll, 0));
    } else {
        e->d_coll = e->d_sl;
    }
    CKC(cudaHostAlloc(&e->h_sstart, CHUNK * 8, cudaHostAllocMapped));
    CKC(cudaHostGetDevicePointer(&e->d_sstart, e->h_sstart, 0));
    CKC(cudaHostAlloc(&e->h_scount, CHUNK * 4, cudaHostAllocMapped));
    CKC(cudaHostGetDevicePointer(&e->d_scount, e->h_scount, 0));
    CKC(cudaMallocHost(&e->h_v, (size_t)e->npad * 4)); CKC(cudaMallocHost(&e->h_g, (size_t)e->npad * 4));
    e->spike_counts = (int32_t *)calloc(n, 4);
    e->spikes = (int32_t *)malloc(4096 * 4);
    pthread_mutex_init(&e->mu, NULL); pthread_cond_init(&e->cv_go, NULL); pthread_cond_init(&e->cv_done, NULL);
    e->done = 1;                       /* no job pending: lif_wait() returns at once */
    e->graph_args = (RunArgs *)calloc(NGRAPH, sizeof(RunArgs));
    /* direct launches, e.g. for Nsight Compute, which does not see a kernel inside a graph */
    e->nograph = getenv("FLY_CUDA_NO_GRAPH") && strcmp(getenv("FLY_CUDA_NO_GRAPH"), "0") != 0;
    lif_reset(e);
    return e;
fail:
    free(pe);
    lif_destroy(e);
    return NULL;
}

static void *driver_main(void *arg);

void lif_destroy(lif *e) {
    if (e->has_th) {
        pthread_mutex_lock(&e->mu); e->quit = 1; pthread_cond_signal(&e->cv_go); pthread_mutex_unlock(&e->mu);
        pthread_join(e->th, NULL);
    }
    cudaStreamSynchronize(e->st);
    for (int k = 0; k < e->ngraph; k++) cudaGraphExecDestroy(e->graph[k]);
    free(e->graph_args);
    void *dp[] = {e->d_st, e->d_indptr, e->d_pe, e->d_wlut, e->d_pmult, e->d_state, e->d_adapt_b,
                  e->d_slow_r, e->d_fl, e->d_rlen, e->d_sil, e->d_pinned, e->d_poi_n, e->d_poi_idx,
                  e->d_ring, e->d_ring_s, e->d_touched, e->d_cnt, e->d_reln, e->d_sl,
                  e->d_tmp_i, e->d_tmp_f, e->d_std_f, e->d_std_d, e->d_std_t};
    for (size_t k = 0; k < sizeof dp / sizeof dp[0]; k++) if (dp[k]) cudaFree(dp[k]);
    cudaFreeHost(e->h_v); cudaFreeHost(e->h_g);
    void *hp[] = {e->h_tot, e->h_coll, e->h_sstart, e->h_scount};
    for (size_t k = 0; k < sizeof hp / sizeof hp[0]; k++) if (hp[k]) cudaFreeHost(hp[k]);
    if (e->h_adapt) cudaFreeHost(e->h_adapt);
    if (e->h_gs) cudaFreeHost(e->h_gs);
    if (e->h_std_d) cudaFreeHost(e->h_std_d);
    if (e->h_std_t) cudaFreeHost(e->h_std_t);
    free(e->wlut); free(e->pmult); free(e->spike_counts); free(e->spikes); free(e->collected);
    free(e->poi_idx); free(e->poi_p); free(e->h_poi_n);
    free(e->hbuf); free(e->sortbits);
    if (e->st) cudaStreamDestroy(e->st);
    free(e);
}

void lif_reset(lif *e) {
    const int n = e->n;
    for (int i = 0; i < e->npad; i++) { e->h_v[i] = -52.0f; e->h_g[i] = 0.0f; }   /* the padding too */
    CK(cudaMemcpy(e->d_v, e->h_v, (size_t)e->npad * 4, cudaMemcpyHostToDevice));
    CK(cudaMemset(e->d_g, 0, (size_t)e->npad * 4));
    CK(cudaMemset(e->d_fl, 0, e->npad));
    CK(cudaMemset(e->d_rlen, e->R, n));
    CK(cudaMemset(e->d_sil, 0, n));
    CK(cudaMemset(e->d_pinned, 0, e->nblk16));
    CK(cudaMemset(e->d_ring, 0, (size_t)(e->D + 1) * n * 8));
    CK(cudaMemset(e->d_touched, 0, (size_t)(e->D + 1) * e->nwords * 4));
    if (e->d_ring_s) CK(cudaMemset(e->d_ring_s, 0, (size_t)(e->D + 1) * n * 8));
    if (e->d_adapt) { CK(cudaMemset(e->d_adapt, 0, (size_t)e->npad * 4)); CK(cudaMemset(e->d_gs, 0, (size_t)e->npad * 4)); }
    if (e->d_std_d) { CK(cudaMemset(e->d_std_d, 0, (size_t)n * 4)); CK(cudaMemset(e->d_std_t, 0, (size_t)n * 8)); }
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
    if (e->npoi)
        k_poi_set<<<nblocks_for(e->npoi, 256), 256, 0, e->st>>>(e->d_poi_idx, e->npoi, e->d_fl,
                                                                 e->d_rlen, e->d_pinned, 0, (uint8_t)e->R);
    CK(cudaMemsetAsync(e->d_pinned, 0, e->nblk16, e->st));
    CK(cudaStreamSynchronize(e->st));                /* before the host buffers are reused */
    if (m > e->poi_cap) {
        if (e->d_poi_idx) cudaFree(e->d_poi_idx);
        e->poi_cap = m * 2 + 1024;
        CK(cudaMalloc(&e->d_poi_idx, (size_t)e->poi_cap * 4));
    }
    e->poi_idx = (int32_t *)realloc(e->poi_idx, (size_t)(m ? m : 1) * 4);
    e->poi_p = (double *)realloc(e->poi_p, (size_t)(m ? m : 1) * 8);
    e->h_poi_n = (int2 *)realloc(e->h_poi_n, (size_t)(m ? m : 1) * 8);
    e->npoi = m;
    e->state_dirty = 1;
    for (int q = 0; q < m; q++) {
        double p = rates_hz[q] * e->dt * 1e-3;
        e->poi_p[q] = p < 0 ? 0 : p > 1 ? 1 : p;
        e->poi_idx[q] = idx[q];
        e->h_poi_n[q] = poi_entry(q, e->poi_p[q]);
    }
    if (m) {
        ensure_tmp(e, m);
        CK(cudaMemcpyAsync(e->d_poi_idx, e->poi_idx, (size_t)m * 4, cudaMemcpyHostToDevice, e->st));
        CK(cudaMemcpyAsync(e->d_tmp_f, e->h_poi_n, (size_t)m * 8, cudaMemcpyHostToDevice, e->st));
        k_poi_n<<<nblocks_for(m, 256), 256, 0, e->st>>>(e->d_poi_idx, (const int2 *)e->d_tmp_f, m, e->d_poi_n);
        k_poi_set<<<nblocks_for(m, 256), 256, 0, e->st>>>(e->d_poi_idx, m, e->d_fl, e->d_rlen,
                                                           e->d_pinned, 1, (uint8_t)e->R);
    }
    CK(cudaStreamSynchronize(e->st));
}

int lif_set_poisson_rates(lif *e, const double *rates_hz, int m) {
    if (m != e->npoi) return -1;
    for (int q = 0; q < m; q++) {
        double p = rates_hz[q] * e->dt * 1e-3;
        e->poi_p[q] = p < 0 ? 0 : p > 1 ? 1 : p;
        e->h_poi_n[q] = poi_entry(q, e->poi_p[q]);
    }
    if (m) {
        ensure_tmp(e, m);
        CK(cudaMemcpyAsync(e->d_tmp_f, e->h_poi_n, (size_t)m * 8, cudaMemcpyHostToDevice, e->st));
        k_poi_n<<<nblocks_for(m, 256), 256, 0, e->st>>>(e->d_poi_idx, (const int2 *)e->d_tmp_f, m, e->d_poi_n);
        CK(cudaStreamSynchronize(e->st));
    }
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
            e->d_adapt = e->d_state + 2 * (size_t)e->npad; e->d_gs = e->d_state + 3 * (size_t)e->npad;
            CK(cudaMalloc(&e->d_adapt_b, (size_t)n * 4));
            CK(cudaMalloc(&e->d_slow_r, (size_t)n * 4));
            CK(cudaMalloc(&e->d_ring_s, (size_t)(e->D + 1) * n * 8));
            CK(cudaMemset(e->d_adapt, 0, (size_t)e->npad * 4)); CK(cudaMemset(e->d_gs, 0, (size_t)e->npad * 4));
            k_fl_and<<<nblocks_for(e->npad, 256), 256, 0, e->st>>>(e->d_fl, e->npad, (uint8_t)~FL_XF);
            CK(cudaStreamSynchronize(e->st));
            CK(cudaMemset(e->d_ring_s, 0, (size_t)(e->D + 1) * n * 8));
            CK(cudaMallocHost(&e->h_adapt, (size_t)n * 4)); CK(cudaMallocHost(&e->h_gs, (size_t)n * 4));
            memset(e->h_adapt, 0, (size_t)n * 4); memset(e->h_gs, 0, (size_t)n * 4);
        }
        CK(cudaMemcpy(e->d_adapt_b, adapt_mV, (size_t)n * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(e->d_slow_r, slow_ratio, (size_t)n * 4, cudaMemcpyHostToDevice));
        e->ea = (float)exp(-e->dt / tau_adapt_ms); e->ka = (float)drive_coef(tau_adapt_ms, e->dt);
        e->es = (float)exp(-e->dt / tau_slow_ms); e->ks = (float)drive_coef(tau_slow_ms, e->dt);
    } else if (e->d_adapt) {
        void *dp[] = {e->d_adapt_b, e->d_slow_r, e->d_ring_s};
        for (int k = 0; k < 3; k++) cudaFree(dp[k]);
        e->d_adapt = e->d_gs = e->d_adapt_b = e->d_slow_r = NULL; e->d_ring_s = NULL;
        cudaFreeHost(e->h_adapt); cudaFreeHost(e->h_gs); e->h_adapt = e->h_gs = NULL;
    }
    e->ext = on;
}

void lif_set_std(lif *e, const float *f, double tau_ms) {
    const int n = e->n;
    int on = 0;
    for (int i = 0; i < n && !on; i++) on = f[i] > 0.0f;
    if (on) {
        if (!e->d_std_f) {
            CK(cudaMalloc(&e->d_std_f, (size_t)n * 4)); CK(cudaMalloc(&e->d_std_d, (size_t)n * 4));
            CK(cudaMalloc(&e->d_std_t, (size_t)n * 8));
            CK(cudaMemset(e->d_std_d, 0, (size_t)n * 4)); CK(cudaMemset(e->d_std_t, 0, (size_t)n * 8));
            CK(cudaMallocHost(&e->h_std_d, (size_t)n * 4)); CK(cudaMallocHost(&e->h_std_t, (size_t)n * 8));
        }
        /* re-enabled after being off: start fully recovered, as the CPU engine
         * (which frees its depletion buffer when depression is switched off) */
        if (!e->std_on) { CK(cudaMemset(e->d_std_d, 0, (size_t)n * 4)); CK(cudaMemset(e->d_std_t, 0, (size_t)n * 8)); }
        CK(cudaMemcpy(e->d_std_f, f, (size_t)n * 4, cudaMemcpyHostToDevice));
        e->std_base = exp(-e->dt / tau_ms);
    }
    e->std_on = on;
}
float *lif_std_depletion(lif *e) {
    /* d as of the last completed step (lazy decay applied to a host copy) */
    if (!e->d_std_d || !e->std_on) return NULL;       /* as the CPU: none when switched off */
    CK(cudaStreamSynchronize(e->st));
    CK(cudaMemcpy(e->h_std_d, e->d_std_d, (size_t)e->n * 4, cudaMemcpyDeviceToHost));
    if (e->std_on) {
        CK(cudaMemcpy(e->h_std_t, e->d_std_t, (size_t)e->n * 8, cudaMemcpyDeviceToHost));
        const long long k = (long long)e->step_count - 1;
        for (int i = 0; i < e->n; i++) {
            const long long gap = k - e->h_std_t[i];
            if (gap > 0 && e->h_std_d[i] != 0.0f)
                e->h_std_d[i] = (float)((double)e->h_std_d[i] * std_decay_d(e->std_base, gap));
        }
    }
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
    /* float additions to one neuron must happen in the CPU's order: with
     * repeated indices, one thread adds them all in order */
    int32_t *srt = (int32_t *)malloc((size_t)m * 4);
    memcpy(srt, idx, (size_t)m * 4);
    qsort(srt, (size_t)m, 4, cmp_i32);
    int dup = 0;
    for (int k = 1; k < m && !dup; k++) dup = srt[k] == srt[k - 1];
    free(srt);
    if (dup) k_add_g_seq<<<1, 1, 0, e->st>>>(e->d_tmp_i, e->d_tmp_f, m, e->d_g);
    else k_add_g<<<nblocks_for(m, 256), 256, 0, e->st>>>(e->d_tmp_i, e->d_tmp_f, m, e->d_g);
    CK(cudaStreamSynchronize(e->st));
    if (e->host_valid) for (int k = 0; k < m; k++) e->h_g[idx[k]] += vals[k];
}

/* run c steps (one cooperative launch; counters live in DevState on the device) */
static void launch_steps(lif *e, int c) {
    RunArgs ra;
    memset(&ra, 0, sizeof ra);                        /* padding too: graphs are matched by memcmp */
    StepArgs &a = ra.a;
    a.n = e->n; a.ext = e->ext;           /* (ext and qtol > 0 also select the kernel) */
    a.ev = e->ev; a.eg = e->eg; a.kg = e->kg; a.v0r = e->v0rest; a.vth = e->vth; a.poi_w = e->poi_w;
    a.vfix = e->vfix; a.qtol = e->qtol; a.ka = e->ka; a.ks = e->ks; a.ea = e->ea; a.es = e->es; a.xe = e->x_eps;
    ra.D = e->D; ra.c = c; ra.nwords = e->nwords; ra.npad = e->npad; ra.lwb = e->lwb;
    ra.cap = (long long)e->coll_cap; ra.vrst = e->vrst; ra.std_base = e->std_base;
    ra.st = e->d_st;
    ra.V = e->d_v; ra.G = e->d_g; ra.A = e->d_adapt; ra.GS = e->d_gs; ra.adapt_b = e->d_adapt_b; ra.slow_r = e->d_slow_r;
    ra.fl = e->d_fl; ra.rlen = e->d_rlen; ra.sil = e->d_sil; ra.pinned = e->d_pinned;
    ra.poi_n = e->d_poi_n;
    ra.ring = e->d_ring; ra.ring_s = e->d_ring_s; ra.touched = e->d_touched;
    ra.indptr = e->d_indptr; ra.pe = e->d_pe; ra.wlut = e->d_wlut; ra.pm = e->plastic ? e->d_pmult : NULL;
    ra.std_f = e->std_on ? e->d_std_f : NULL; ra.std_d = e->d_std_d; ra.std_t = e->d_std_t;
    ra.cnt = e->d_cnt; ra.reln = e->d_reln; ra.sl = e->d_sl;
    ra.tot = e->d_tot; ra.coll = e->d_coll; ra.sstart = e->d_sstart; ra.scount = e->d_scount;
    /* A cooperative launch costs ~50 us of host time on a Jetson; a CUDA
     * graph of it ~10 us. Graphs are cached by their exact arguments (any
     * change of pointers, constants or step count captures a new one). */
    for (int k = 0; k < e->ngraph; k++)
        if (!memcmp(&e->graph_args[k], &ra, sizeof ra)) { CK(cudaGraphLaunch(e->graph[k], e->st)); return; }
    void *args[] = {&ra};
    if (e->ngraph == NGRAPH) {                       /* full: replace the oldest */
        cudaGraphExecDestroy(e->graph[0]);
        memmove(e->graph, e->graph + 1, (NGRAPH - 1) * sizeof e->graph[0]);
        memmove(e->graph_args, e->graph_args + 1, (NGRAPH - 1) * sizeof e->graph_args[0]);
        e->ngraph--;
    }
    const void *kern = run_kernel(e->ext, e->qtol > 0.0f, e->smem > 0);
    /* capture; if that fails (cooperative launches in graphs need CUDA 12,
     * or another thread's legacy-stream call invalidated the capture),
     * launch directly from now on */
    cudaGraph_t g = NULL;
    cudaGraphExec_t ex = NULL;
    int ok = !e->nograph && cudaStreamBeginCapture(e->st, cudaStreamCaptureModeThreadLocal) == cudaSuccess;
    if (ok) {
        const cudaError_t e1 = cudaMemsetAsync(e->d_cnt, 0, (size_t)c * 4, e->st);
        const cudaError_t e2 = cudaLaunchCooperativeKernel(kern, dim3(e->nbk), dim3(RB), args, (size_t)e->smem, e->st);
        const cudaError_t e3 = cudaStreamEndCapture(e->st, &g);
        ok = e1 == cudaSuccess && e2 == cudaSuccess && e3 == cudaSuccess && g && cudaGraphInstantiate(&ex, g, 0) == cudaSuccess;
        if (g) cudaGraphDestroy(g);
    }
    if (!ok) {
        cudaGetLastError();
        e->nograph = 1;
        CK(cudaMemsetAsync(e->d_cnt, 0, (size_t)c * 4, e->st));
        CK(cudaLaunchCooperativeKernel(kern, dim3(e->nbk), dim3(RB), args, (size_t)e->smem, e->st));
        return;
    }
    e->graph[e->ngraph] = ex; e->graph_args[e->ngraph] = ra; e->ngraph++;
    CK(cudaGraphLaunch(ex, e->st));
}

/* host mirror of the device counters after `c` steps */
static void advance_host(lif *e, int c) {
    e->rng += 0x6D2B79F5u * (uint32_t)e->npoi * (uint32_t)c;
    e->slot = (e->slot + c) % (e->D + 1);
    e->step_count += c;
}

/* push host counters to the device state (after reset / seed / poisson changes) */
static void push_state(lif *e) {
    DevState h; memset(&h, 0, sizeof h);
    h.slot = e->slot; h.npoi = e->npoi; h.rng = e->rng; h.step = e->step_count;
    CK(cudaMemcpyAsync(e->d_st, &h, sizeof h, cudaMemcpyHostToDevice, e->st));
    CK(cudaStreamSynchronize(e->st));
}

/* one step's spikes into ascending order (a neuron spikes at most once a step) */
static void sort_step(lif *e, int32_t *a, int m) {
    if (m > 64) {                                    /* many: through a bitmap of the neurons */
        if (!e->sortbits) e->sortbits = (uint32_t *)calloc(e->nwords, 4);
        uint32_t *bm = e->sortbits;
        for (int k = 0; k < m; k++) bm[a[k] >> 5] |= 1u << (a[k] & 31);
        int o = 0;
        for (int w = 0; w < e->nwords && o < m; w++) {
            uint32_t x = bm[w];
            bm[w] = 0;                               /* left clear for the next step */
            while (x) { a[o++] = w * 32 + __builtin_ctz(x); x &= x - 1; }
        }
        return;
    }
    for (int i = 1; i < m; i++) {                    /* few: insertion sort */
        const int32_t x = a[i];
        int k = i - 1;
        while (k >= 0 && a[k] > x) { a[k + 1] = a[k]; k--; }
        a[k + 1] = x;
    }
}

static void collect_host(lif *e, const int32_t *src, long k) {
    if (e->ncollected + k > e->collected_cap) {
        e->collected_cap = (e->ncollected + k) * 2 + 4096;
        e->collected = (int32_t *)realloc(e->collected, (size_t)e->collected_cap * 4);
    }
    memcpy(e->collected + e->ncollected, src, (size_t)k * 4);
    e->ncollected += k;
}

/* the longest chunk whose worst-case spike count fits the buffer (see lif_create) */
static int chunk_steps(const lif *e) {
    const long long np = e->npoi, other = (long long)e->n - np;
    long long c = e->watchdog ? 100 : CHUNK;          /* a display GPU's time limit: short launches */
    while (c > 1 && np * c + other * (c / e->R + 1) > e->coll_cap) c = c * 9 / 10;
    return (int)c;
}

/* run `steps` steps; collect=1 keeps every spike in e->collected */
static pthread_mutex_t g_run_mu = PTHREAD_MUTEX_INITIALIZER;

static long run_steps(lif *e, int steps, int collect) {
    if (e->plastic && e->pm_dirty) {
        CK(cudaMemcpy(e->d_pmult, e->pmult, (size_t)e->nnz * 4, cudaMemcpyHostToDevice));
        e->pm_dirty = 0;
    }
    if (e->state_dirty) { push_state(e); e->state_dirty = 0; }
    e->ncollected = 0;
    long total = 0;
    for (int s = 0; s < steps;) {
        const int cmax = chunk_steps(e);
        const int c = steps - s < cmax ? steps - s : cmax;
        /* k_run needs every block resident at once and fills the GPU: two
         * engines in one process must not run at the same time */
        pthread_mutex_lock(&g_run_mu);
        launch_steps(e, c);
        advance_host(e, c);
        CK(cudaStreamSynchronize(e->st));
        pthread_mutex_unlock(&g_run_mu);
        const long long tot = *(volatile long long *)e->h_tot;
        if (tot > e->coll_cap) { fprintf(stderr, "lif_cuda: spike buffer overflow (%lld)\n", tot); abort(); }
        if (tot > e->hcap) { e->hcap = tot * 2 + 4096; e->hbuf = (int32_t *)realloc(e->hbuf, (size_t)e->hcap * 4); }
        int32_t *hbuf = e->hbuf;
        if (tot) {
            if (e->h_coll) memcpy(hbuf, e->h_coll, (size_t)tot * 4);
            else CK(cudaMemcpy(hbuf, e->d_coll, (size_t)tot * 4, cudaMemcpyDeviceToHost));
        }
        const long long *hstart = e->h_sstart; const int *hcount = e->h_scount;
        for (int k = 0; k < c; k++) sort_step(e, hbuf + hstart[k], hcount[k]);   /* each step ascending, as the CPU */
        for (long q = 0; q < tot; q++) e->spike_counts[hbuf[q]]++;
        if (collect) collect_host(e, hbuf, tot);
        /* lif_spikes(): the last step's spikes */
        const int nl = hcount[c - 1];
        e->spikes = (int32_t *)realloc(e->spikes, (size_t)(nl ? nl : 1) * 4);
        if (nl) memcpy(e->spikes, hbuf + hstart[c - 1], (size_t)nl * 4);
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
