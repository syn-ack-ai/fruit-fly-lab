/*
 * Native whole-brain LIF engine (Shiu et al. 2024) for aarch64 / Raspberry Pi 5.
 *
 * Bit-exact with simulation/engine/lif_engine.py and web/js/engine.js: every
 * float32 operation happens in the same order with the same rounding, the
 * compiler is forbidden from fusing multiply-adds (-ffp-contract=off), and
 * Poisson draws use the shared mulberry32 stream in the same order.
 * native/verify_native.py checks this state-for-state.
 *
 * What differs is only how the work is laid out:
 *
 *  1. Sparse delay line. The reference ring buffer writes step t's output into
 *     slot (t-1) mod 19, which was delivered and zeroed on the previous step,
 *     so every write lands on zeros: ring[j] = fround(0 + acc[j]) = (float)acc[j].
 *     The dense 19 x 139,255 ring becomes one sparse (target, value) list per slot.
 *
 *  2. Order-independent accumulation. Synaptic weights are float32 values
 *     fround(count * 0.275) whose bits lie within 2^-25 .. 2^17, so their
 *     float64 sum is exact in any order (as np.bincount's is). That lets each
 *     thread sum its own targets independently.
 *
 *  3. Owner computes. Neurons are split into contiguous ranges, one per core,
 *     small enough to stay resident in that A76's L2. A thread only ever writes
 *     state for neurons it owns: it integrates them, delivers events to them,
 *     and scatters the part of each spiking row (rows are sorted) that lands in
 *     its range into its own delay line. Two spin barriers per step. The
 *     caller is worker 0 and is pinned to core 0; workers take cores 1..T-1.
 *     Per-step accumulation uses a small per-thread hash table rather than a
 *     full-size array, which would evict the thread's v/g slice from L2.
 *
 *  4. Quiescent blocks. A 16-neuron block whose update leaves every v and g
 *     bitwise unchanged, with none refractory, is at a fixed point of the
 *     float32 map and would stay there in the reference engine too. It is
 *     skipped until an event is delivered to it or one of its neurons spikes.
 *     Blocks holding Poisson targets are never skipped.
 */
#define _GNU_SOURCE
#if defined(__aarch64__) && !defined(LIF_GENERIC)
#include <arm_neon.h>
#define CPU_RELAX() __asm__ volatile("yield")
#elif defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define CPU_RELAX() _mm_pause()
#else
#define CPU_RELAX() ((void)0)
#endif
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#if defined(__linux__)
#include <linux/futex.h>
#include <sys/syscall.h>
#elif defined(__APPLE__)
#include <os/os_sync_wait_on_address.h>
#endif

/* ------------------------------------------------ platform: sleep & pinning */
/* Block while *addr == val (spurious wake-ups are fine; callers re-check). */
static void futex_wait(_Atomic int *addr, int val) {
#if defined(__linux__)
    syscall(SYS_futex, addr, FUTEX_WAIT_PRIVATE, val, NULL, NULL, 0);
#elif defined(__APPLE__)
    os_sync_wait_on_address((void *)addr, (uint64_t)(uint32_t)val, sizeof(int),
                            OS_SYNC_WAIT_ON_ADDRESS_NONE);
#else
    (void)addr; (void)val; sched_yield();
#endif
}
static void futex_wake(_Atomic int *addr) {
#if defined(__linux__)
    syscall(SYS_futex, addr, FUTEX_WAKE_PRIVATE, INT32_MAX, NULL, NULL, 0);
#elif defined(__APPLE__)
    os_sync_wake_by_address_all((void *)addr, sizeof(int), OS_SYNC_WAKE_BY_ADDRESS_NONE);
#else
    (void)addr;
#endif
}
/* Pin a thread to one core. Linux only; macOS does not allow pinning and its
 * scheduler keeps busy threads on the performance cores anyway. */
static void pin_thread(pthread_t th, int cpu) {
#if defined(__linux__)
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(cpu, &cs);
    pthread_setaffinity_np(th, sizeof cs, &cs);
#else
    (void)th; (void)cpu;
#endif
}

static inline double now_us(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec * 1e-3;
}

#define MAXT 8
#define BLK 16

typedef struct { int32_t *idx; float *val, *val_s; int n, cap; } evlist;   /* val_s: slow inhibition */
/* One connection in 4 bytes: target neuron in the high 18 bits, signed synapse
 * count + 8192 in the low 14. The weight in mV comes from a lookup table built
 * with the same arithmetic as before, so results are bit-identical, and the
 * connection list takes half the memory bandwidth (it dominates on the Pi). */
#define PE_SHIFT 14
#define PE_MASK ((1u << PE_SHIFT) - 1)
#define PE_BIAS 8192
#define PE_MAX_N (1 << (32 - PE_SHIFT))

typedef struct lif lif;

typedef struct {
    lif *e;
    int tid, lo, hi;                 /* owned neurons; lo, hi multiples of 64 except final hi */
    int32_t *spk; int nspk, cap;     /* phase A: non-Poisson spikes in [lo,hi), ascending */
    /* phase B: per-step accumulator, open addressing on target index. Small
     * enough to stay in L1/L2, unlike a full-size array which would evict
     * this thread's slice of v and g every step. */
    int32_t *hkey; double *hval, *hval_s; int hbits;
    int32_t *hused; int nused;
    evlist *ring;                    /* D+1 sparse delay slots for owned targets */
    int32_t *rfc; int nrfc, rfc_cap;  /* owned neurons currently refractory */
    int32_t *xl; int nxl, xcap;      /* owned neurons with adaptation / slow inhibition != 0 */
    float *hold_v, *hold_g;
    long active_blocks, edges;        /* statistics */
    double busy_a, busy_b, n_deliv, t_deliv;
    pthread_t th;
    char pad[64];
} worker;

struct lif {
    int n, nnz, D, R, nthreads, nblocks;
    int cpu_base, ncpu;              /* this engine pins to cores cpu_base.. */
    const int32_t *indptr, *indices;
    uint32_t *pe;                    /* CSR data, packed (see PE_SHIFT) */
    float *wlut;                     /* synapse count + PE_BIAS -> weight in mV */
    float *pmult;                    /* optional per-connection multiplier (plasticity); NULL = off */
    int32_t *split;                  /* [n][nthreads+1]: where each row enters each thread's range */

    float *v, *g;
    uint8_t *rfc_left, *rfc_len, *silenced, *is_poi;
    uint8_t *quiet, *pinned;         /* per block */
    int32_t *spike_counts;

    int slot;
    float ev, eg, kg, v0rest, vth, vrst, poi_w;
    uint32_t rng;
    int32_t *poi_idx; double *poi_p; int npoi;

    int32_t *spikes; int nspikes, spikes_cap;
    long step_count;
    double t_a, t_serial, t_b;       /* cumulative microseconds per phase */

    worker wk[MAXT];
    _Atomic int go_gen __attribute__((aligned(64)));
    _Atomic int phase;
    _Atomic int done_count __attribute__((aligned(64)));
    _Atomic int quit;
    _Atomic int sleepers;

    int32_t *collected; long ncollected, collected_cap;   /* lif_run_collect output */

    /* Optional dynamics beyond the published model (off unless lif_set_dynamics
     * enables them; the published code paths are untouched when off):
     *   adaptation  per-neuron hyperpolarising drive, +adapt_b mV per spike,
     *               decaying with tau_a (spike-frequency adaptation)
     *   slow inhib. every inhibitory synapse also drives a slow channel gs
     *               (GABA-B-like), slow_ratio x its weight, decaying with tau_s
     * Both enter dv/dt linearly, so integration stays exact. */
    int ext;
    float *adapt, *gs;
    float ea, ka, es, ks;
    float *adapt_b;                  /* per neuron: adaptation added per spike (mV) */
    uint8_t *xflag;                  /* neuron is on its owner's extra-state list */
    float *slow_r;                   /* per PRESYNAPTIC neuron: slow share of its inhibition */

    /* Optional approximate quiescence (lif_set_quiesce_tol; 0 = exact, the
     * default): a block whose neurons are all within qtol mV of the resting
     * fixed point and have |g| < qtol is snapped to rest and skipped. */
    float qtol, vfix, x_eps;
    double dt;                       /* ms; 0.1 = Shiu et al. 2024 (FLY_DT overrides) */

    /* Optional short-term depression of a neuron's OUTPUT synapses (lif_set_std;
     * off unless enabled): depletion d in [0,1) per presynaptic neuron, decaying
     * by std_e every step; a spike releases (1 - d) of full strength and leaves
     * d = 1 - (1 - d) * f (f = fraction of release sites still available). */
    float *std_f, *std_d, *std_rel; float std_e;
    int32_t *std_list; int std_n;

    /* asynchronous driver: runs lif_run_collect on its own thread */
    pthread_t driver; int has_driver;
    _Atomic int job_gen __attribute__((aligned(64)));
    _Atomic int job_done __attribute__((aligned(64)));
    _Atomic int job_steps, driver_sleeping, waiter_sleeping;
};

/* ------------------------------------------------------------------ PRNG */
static inline double mulberry32(uint32_t *s) {
    uint32_t a = (*s += 0x6D2B79F5u);
    uint32_t t = (uint32_t)((a ^ (a >> 15)) * (1u | a));
    t = (t + (uint32_t)((t ^ (t >> 7)) * (61u | t))) ^ t;
    return (double)(t ^ (t >> 14)) / 4294967296.0;
}

/* ---------------------------------------------------------------- helpers */
static void ev_push(evlist *l, int32_t j, float x, float xs) {
    if (l->n == l->cap) {
        l->cap = l->cap ? l->cap * 2 : 1024;
        l->idx = realloc(l->idx, (size_t)l->cap * sizeof(int32_t));
        l->val = realloc(l->val, (size_t)l->cap * sizeof(float));
        l->val_s = realloc(l->val_s, (size_t)l->cap * sizeof(float));
    }
    l->idx[l->n] = j; l->val[l->n] = x; l->val_s[l->n] = xs; l->n++;
}

static void push_spike(lif *e, int32_t i) {
    if (e->nspikes == e->spikes_cap) {
        e->spikes_cap *= 2;
        e->spikes = realloc(e->spikes, (size_t)e->spikes_cap * sizeof(int32_t));
    }
    e->spikes[e->nspikes++] = i;
}

static int cmp_i32(const void *a, const void *b) {
    int32_t x = *(const int32_t *)a, y = *(const int32_t *)b;
    return (x > y) - (x < y);
}

/* first position in sorted a[lo..hi) with a[pos] >= key */
static inline int32_t lower_bound(const int32_t *a, int32_t lo, int32_t hi, int32_t key) {
    while (lo < hi) {
        int32_t mid = lo + ((hi - lo) >> 1);
        if (a[mid] < key) lo = mid + 1; else hi = mid;
    }
    return lo;
}

/* --------------------------------------------- phase A: deliver + integrate */
#define QUIESCE_EVERY 32
static int debug_noquiet;

/* One 16-neuron block of exact linear integration. Returns whether every v
 * and g is bitwise unchanged (only computed when `check`); *vmax_out is the
 * largest new v, a cheap "anything near threshold" hint. Both variants do the
 * same IEEE float32 operations in the same order (no fused multiply-add), so
 * results are identical on every architecture. */
#if defined(__aarch64__) && !defined(LIF_GENERIC)
static inline int block_update(float *restrict v, float *restrict g, int i,
                               float evs, float egs, float kgs, float v0rs,
                               int check, float *vmax_out) {
    const float32x4_t ev = vdupq_n_f32(evs), eg = vdupq_n_f32(egs);
    const float32x4_t kg = vdupq_n_f32(kgs), v0r = vdupq_n_f32(v0rs);
    float32x4_t vmax = vdupq_n_f32(-INFINITY);
    uint32x4_t same = vdupq_n_u32(~0u);
    for (int q = 0; q < 4; q++) {
        float32x4_t gi = vld1q_f32(g + i + 4 * q), vi = vld1q_f32(v + i + 4 * q);
        float32x4_t tmp = vmulq_f32(gi, kg);
        float32x4_t nv = vaddq_f32(vaddq_f32(vmulq_f32(vi, ev), v0r), tmp);
        float32x4_t ng = vmulq_f32(gi, eg);
        vst1q_f32(v + i + 4 * q, nv);
        vst1q_f32(g + i + 4 * q, ng);
        vmax = vmaxq_f32(vmax, nv);
        if (check) {
            same = vandq_u32(same, vceqq_u32(vreinterpretq_u32_f32(nv), vreinterpretq_u32_f32(vi)));
            same = vandq_u32(same, vceqq_u32(vreinterpretq_u32_f32(ng), vreinterpretq_u32_f32(gi)));
        }
    }
    *vmax_out = vmaxvq_f32(vmax);
    return check && vminvq_u32(same) != 0;
}
#else
static inline int block_update(float *restrict v, float *restrict g, int i,
                               float ev, float eg, float kg, float v0r,
                               int check, float *vmax_out) {
    float vmax = -INFINITY;
    uint32_t diff = 0;
    for (int j = i; j < i + BLK; j++) {          /* auto-vectorised (AVX2 on x86) */
        float gi = g[j], vi = v[j];
        float tmp = gi * kg;
        float nv = vi * ev; nv = nv + v0r; nv = nv + tmp;
        float ng = gi * eg;
        v[j] = nv; g[j] = ng;
        vmax = nv > vmax ? nv : vmax;
        uint32_t a, b, c, d;
        memcpy(&a, &nv, 4); memcpy(&b, &vi, 4); memcpy(&c, &ng, 4); memcpy(&d, &gi, 4);
        diff |= (a ^ b) | (c ^ d);
    }
    *vmax_out = vmax;
    return check && diff == 0;
}
#endif

/* Extended dynamics are applied sparsely: the dense update above stays the
 * published one, and neurons carrying adaptation or slow inhibition get the
 * extra terms afterwards (same float order: nv - a*ka + gs*ks, with the old a
 * and gs), then a and gs decay. Values below X_EPS mV are flushed to zero and
 * the neuron leaves the list, so settled regions can go quiet again. */
#ifndef X_EPS
#define X_EPS 1e-5f
#endif

static inline void x_mark(lif *e, worker *k, int32_t j) {
    if (e->xflag[j]) return;
    e->xflag[j] = 1;
    if (k->nxl == k->xcap) {
        k->xcap = k->xcap ? 2 * k->xcap : 4096;
        k->xl = realloc(k->xl, (size_t)k->xcap * sizeof(int32_t));
    }
    k->xl[k->nxl++] = j;
}

static void phase_a(lif *e, worker *k) {
    float *restrict v = e->v, *restrict g = e->g;
    uint8_t *restrict rl = e->rfc_left, *restrict quiet = e->quiet;
    const uint8_t *restrict poi = e->is_poi, *restrict pinned = e->pinned;

    /* 1. deliver the events scheduled D steps ago (owned targets only) */
    double td = now_us();
    evlist *in = &k->ring[e->slot];
    k->n_deliv += in->n;
    for (int q = 0; q < in->n; q++) {
        int32_t j = in->idx[q];
        g[j] += in->val[q];
        quiet[j / BLK] = 0;
    }
    if (e->ext)
        for (int q = 0; q < in->n; q++)
            if (in->val_s[q] != 0.0f) { e->gs[in->idx[q]] += in->val_s[q]; x_mark(e, k, in->idx[q]); }
    in->n = 0;
    k->t_deliv += now_us() - td;

    /* 2. Brian2 `(unless refractory)`: refractory neurons are not integrated.
     * They are few, so integrate everything and put them back afterwards. */
    for (int q = 0; q < k->nrfc; q++) {
        int32_t j = k->rfc[q];
        k->hold_v[q] = v[j]; k->hold_g[q] = g[j];
    }

    /* 3. exact linear integration, 16 neurons per block (block_update), with a
     * cheap "anything near threshold" hint. Every QUIESCE_EVERY steps, blocks
     * whose state is bitwise unchanged are marked quiet (fixed point). */
    const float ev = e->ev, eg = e->eg, kg = e->kg, v0r = e->v0rest;
    const float vth = e->vth;
    const int check = (e->step_count % QUIESCE_EVERY) == 0;
    k->nspk = 0;
    long active = 0;

    int i = k->lo;
    for (; i + BLK <= k->hi; i += BLK) {
        int b = i / BLK;
        if (quiet[b]) continue;
        active++;
        float vmax;
        int same = block_update(v, g, i, ev, eg, kg, v0r, check, &vmax);
        if (__builtin_expect(vmax > vth, 0))
            for (int j = i; j < i + BLK; j++)
                if (v[j] > vth && !poi[j] && !rl[j]) {
                    if (k->nspk == k->cap) {
                        k->cap *= 2;
                        k->spk = realloc(k->spk, (size_t)k->cap * sizeof(int32_t));
                    }
                    k->spk[k->nspk++] = j;
                }
        if (!same && check && e->qtol > 0.0f && vmax < vth && !pinned[b]) {
            const float vf = e->vfix, tol = e->qtol;
            int near = 1;
            for (int j = i; j < i + BLK && near; j++)
                near = fabsf(v[j] - vf) <= tol && fabsf(g[j]) <= tol && !rl[j];
            if (near) {
                for (int j = i; j < i + BLK; j++) { v[j] = vf; g[j] = 0.0f; }
                same = 1;
            }
        }
        if (same && !pinned[b] && !debug_noquiet) quiet[b] = 1;
    }
    for (; i < k->hi; i++) {                               /* unaligned tail */
        float gi = g[i];
        float tmp = gi * e->kg;
        float nv = v[i] * e->ev; nv = nv + e->v0rest; nv = nv + tmp;
        v[i] = nv; g[i] = gi * e->eg;
        if (nv > vth && !poi[i] && !rl[i]) {
            if (k->nspk == k->cap) {
                k->cap *= 2;
                k->spk = realloc(k->spk, (size_t)k->cap * sizeof(int32_t));
            }
            k->spk[k->nspk++] = i;
        }
    }

    /* 3b. extended dynamics on the neurons that carry them (see x_mark) */
    if (e->ext && k->nxl) {
        float *restrict a = e->adapt, *restrict gs = e->gs;
        const float ka = e->ka, ks = e->ks, ea = e->ea, es = e->es;
        int keep = 0;
        for (int q = 0; q < k->nxl; q++) {
            const int32_t j = k->xl[q];
            float nv = v[j] - a[j] * ka;
            nv = nv + gs[j] * ks;
            v[j] = nv;
            a[j] *= ea; gs[j] *= es;
            quiet[j / BLK] = 0;
            if (a[j] < e->x_eps && gs[j] > -e->x_eps) { a[j] = 0.0f; gs[j] = 0.0f; e->xflag[j] = 0; }
            else k->xl[keep++] = j;
        }
        k->nxl = keep;
        /* the extra terms only lower v: drop candidates now below threshold */
        int ns = 0;
        for (int q = 0; q < k->nspk; q++)
            if (v[k->spk[q]] > vth) k->spk[ns++] = k->spk[q];
        k->nspk = ns;
    }

    /* 4. restore refractory neurons, count them down, keep their blocks awake */
    int keep = 0;
    for (int q = 0; q < k->nrfc; q++) {
        int32_t j = k->rfc[q];
        v[j] = k->hold_v[q]; g[j] = k->hold_g[q];
        quiet[j / BLK] = 0;
        if (--rl[j]) k->rfc[keep++] = j;
    }
    k->nrfc = keep;
    k->active_blocks += active;
}

/* ------------------------------------- phase B: reset + scatter owned targets */
#define HEMPTY (-1)

static inline uint32_t hslot(int32_t j, int bits) {
    return ((uint32_t)j * 2654435761u) >> (32 - bits);
}

static void hash_init(worker *k, int bits) {
    k->hbits = bits;
    size_t sz = (size_t)1 << bits;
    k->hkey = malloc(sz * sizeof(int32_t));
    k->hval = malloc(sz * sizeof(double));
    k->hval_s = calloc(sz, sizeof(double));
    k->hused = malloc(sz * sizeof(int32_t));
    memset(k->hkey, 0xFF, sz * sizeof(int32_t));
    k->nused = 0;
}

static void hash_free(worker *k) { free(k->hkey); free(k->hval); free(k->hval_s); free(k->hused); }

static void hash_grow(worker *k) {
    int32_t *ok = k->hkey; double *ov = k->hval, *os = k->hval_s; int32_t *ou = k->hused; int on = k->nused;
    hash_init(k, k->hbits + 1);
    for (int u = 0; u < on; u++) {                       /* keep first-touch order */
        uint32_t m = (1u << k->hbits) - 1, h = hslot(ok[ou[u]], k->hbits);
        while (k->hkey[h] != HEMPTY) h = (h + 1) & m;
        k->hkey[h] = ok[ou[u]]; k->hval[h] = ov[ou[u]]; k->hval_s[h] = os[ou[u]];
        k->hused[k->nused++] = (int32_t)h;
    }
    free(ok); free(ov); free(os); free(ou);
}

static void phase_b(lif *e, worker *k) {
    const uint32_t *restrict pe = e->pe;
    const float *restrict wlut = e->wlut;
    const float *restrict pm = e->pmult;
    const int32_t *restrict split = e->split;
    const int32_t *restrict spikes = e->spikes;
    const int T1 = e->nthreads + 1, tid = k->tid, ns = e->nspikes;
    const int lo = k->lo, hi = k->hi;

    /* rows are scattered over the packed connection list: start every DRAM fetch first */
    for (int s = 0; s < ns; s++) __builtin_prefetch(&split[(long)spikes[s] * T1 + tid]);
    for (int s = 0; s < ns; s++) __builtin_prefetch(&pe[split[(long)spikes[s] * T1 + tid]]);

    for (int s = 0; s < ns; s++) {
        int32_t i = spikes[s];
        if (i >= lo && i < hi) {                          /* reference reset: v=v_rst, g=0 */
            e->v[i] = e->vrst; e->g[i] = 0.0f;
            e->spike_counts[i]++;
            e->quiet[i / BLK] = 0;
            if (e->ext && e->adapt_b[i] != 0.0f) { e->adapt[i] += e->adapt_b[i]; x_mark(e, k, i); }
            if (e->rfc_len[i] && !e->rfc_left[i]) {
                if (k->nrfc == k->rfc_cap) {
                    k->rfc_cap *= 2;
                    k->rfc = realloc(k->rfc, (size_t)k->rfc_cap * sizeof(int32_t));
                    k->hold_v = realloc(k->hold_v, (size_t)k->rfc_cap * sizeof(float));
                    k->hold_g = realloc(k->hold_g, (size_t)k->rfc_cap * sizeof(float));
                }
                k->rfc[k->nrfc++] = i;
            }
            e->rfc_left[i] = e->rfc_len[i];
        }
        if (e->silenced[i]) continue;
        const int32_t *sp = &split[(long)i * T1 + tid];   /* rows are sorted by target */
        int32_t a = sp[0], z = sp[1];
        const float rel = (e->std_n && e->std_f[i] > 0.0f) ? e->std_rel[i] : 1.0f;
        k->edges += z - a;
        for (int32_t q = a; q < z; q++) {
            const uint32_t p = pe[q];
            int32_t j = (int32_t)(p >> PE_SHIFT);
            uint32_t m = (1u << k->hbits) - 1, h = hslot(j, k->hbits);
            while (k->hkey[h] != j && k->hkey[h] != HEMPTY) h = (h + 1) & m;
            if (k->hkey[h] == HEMPTY) {
                k->hkey[h] = j; k->hval[h] = 0.0; k->hval_s[h] = 0.0;
                k->hused[k->nused++] = (int32_t)h;
                if (2 * k->nused > (1 << k->hbits)) {    /* keep load under 50% */
                    hash_grow(k);
                    m = (1u << k->hbits) - 1; h = hslot(j, k->hbits);
                    while (k->hkey[h] != j) h = (h + 1) & m;
                }
            }
            /* exact float64 sum; with plasticity on, each weight is scaled first
             * (multipliers start at 1.0f, so an untrained brain is unchanged) */
            float w = pm ? wlut[p & PE_MASK] * pm[q] : wlut[p & PE_MASK];
            if (rel != 1.0f) w = w * rel;
            k->hval[h] += w;
            if (e->ext && w < 0.0f) k->hval_s[h] += w * e->slow_r[i];   /* inhibitory: slow channel */
        }
    }
    evlist *out = &k->ring[(e->slot + e->D) % (e->D + 1)];
    for (int u = 0; u < k->nused; u++) {
        int32_t h = k->hused[u];
        if (k->hval[h] != 0.0 || k->hval_s[h] != 0.0)
            ev_push(out, k->hkey[h], (float)k->hval[h], (float)k->hval_s[h]);
        k->hkey[h] = HEMPTY;
    }
    k->nused = 0;
}

/* ------------------------------------------------------------ thread pool */
/* Threads spin for up to SPIN_US between jobs (a step is tens of
 * microseconds, and a real-time loop hands over work every millisecond), then
 * sleep on a futex so an idle or paused simulation costs no CPU. go_gen and
 * sleepers are both sequentially consistent, so a wake is never lost: either
 * the worker sees the new generation, or the main thread sees the sleeper. */
#define SPIN_US 2000.0

static inline int keep_spinning(int *spins, double *t0) {
    if (++*spins & 255) { CPU_RELAX(); return 1; }
    double t = now_us();
    if (*t0 == 0) { *t0 = t; return 1; }
    return t - *t0 < SPIN_US;
}

static void *worker_main(void *arg) {
    worker *k = arg;
    lif *e = k->e;
    int seen = 0;
    for (;;) {
        int gen, spins = 0; double ts = 0;
        while ((gen = atomic_load(&e->go_gen)) == seen) {
            if (keep_spinning(&spins, &ts)) continue;
            atomic_fetch_add(&e->sleepers, 1);
            if (atomic_load(&e->go_gen) == seen)
                futex_wait(&e->go_gen, seen);
            atomic_fetch_sub(&e->sleepers, 1);
            spins = 0; ts = 0;
        }
        seen = gen;
        if (atomic_load_explicit(&e->quit, memory_order_relaxed)) return NULL;
        double t0 = now_us();
        if (atomic_load_explicit(&e->phase, memory_order_relaxed) == 0) { phase_a(e, k); k->busy_a += now_us() - t0; }
        else { phase_b(e, k); k->busy_b += now_us() - t0; }
        atomic_fetch_add_explicit(&e->done_count, 1, memory_order_acq_rel);
    }
}

static void wake_workers(lif *e) {
    atomic_fetch_add(&e->go_gen, 1);
    if (atomic_load(&e->sleepers))
        futex_wake(&e->go_gen);
}

static void run_phase(lif *e, int ph) {
    if (e->nthreads > 1) {
        atomic_store_explicit(&e->done_count, 0, memory_order_relaxed);
        atomic_store_explicit(&e->phase, ph, memory_order_relaxed);
        wake_workers(e);
    }
    double t0 = now_us();
    if (ph == 0) { phase_a(e, &e->wk[0]); e->wk[0].busy_a += now_us() - t0; }   /* caller = worker 0 */
    else { phase_b(e, &e->wk[0]); e->wk[0].busy_b += now_us() - t0; }
    if (e->nthreads > 1)
        while (atomic_load_explicit(&e->done_count, memory_order_acquire) != e->nthreads - 1)
            CPU_RELAX();
}

/* ------------------------------------------------------------------ API */
void lif_wake_all(lif *e) { memset(e->quiet, 0, e->nblocks); }

void lif_reset(lif *e) {
    int n = e->n;
    for (int i = 0; i < n; i++) { e->v[i] = -52.0f; e->g[i] = 0.0f; }
    memset(e->rfc_left, 0, n);
    memset(e->rfc_len, e->R, n);
    memset(e->silenced, 0, n);
    memset(e->is_poi, 0, n);
    memset(e->pinned, 0, e->nblocks);
    lif_wake_all(e);
    memset(e->spike_counts, 0, (size_t)n * sizeof(int32_t));
    if (e->adapt) memset(e->adapt, 0, (size_t)n * sizeof(float));
    if (e->gs) memset(e->gs, 0, (size_t)n * sizeof(float));
    if (e->xflag) memset(e->xflag, 0, n);
    if (e->std_d) memset(e->std_d, 0, (size_t)n * sizeof(float));
    for (int t = 0; t < e->nthreads; t++) e->wk[t].nxl = 0;
    for (int t = 0; t < e->nthreads; t++) {
        for (int s = 0; s <= e->D; s++) e->wk[t].ring[s].n = 0;
        e->wk[t].active_blocks = 0; e->wk[t].edges = 0;
        e->wk[t].busy_a = e->wk[t].busy_b = e->wk[t].n_deliv = e->wk[t].t_deliv = 0; e->wk[t].nrfc = 0;
    }
    e->slot = 0; e->step_count = 0; e->nspikes = 0; e->npoi = 0;
    e->t_a = e->t_serial = e->t_b = 0;
}

void lif_debug_noquiet(int on) { debug_noquiet = on; }

/* Approximate quiescence (see the lif struct); tol_mV = 0 restores exact mode.
 * Also flushes adaptation / slow inhibition below max(1e-5, 10*tol) mV. */
void lif_set_quiesce_tol(lif *e, float tol_mV) {
    e->qtol = tol_mV > 0.0f ? tol_mV : 0.0f;
    e->x_eps = e->qtol > 0.0f ? fmaxf(X_EPS, 10.0f * e->qtol) : X_EPS;
}

lif *lif_create(int n, int nnz, const int32_t *indptr, const int32_t *indices,
                const int16_t *weights, uint32_t seed, int nthreads) {
    lif *e = aligned_alloc(64, (sizeof(lif) + 63) & ~(size_t)63);
    memset(e, 0, sizeof *e);
    e->n = n; e->nnz = nnz; e->indptr = indptr; e->indices = indices;
    e->nblocks = (n + BLK - 1) / BLK;
    /* dt: the published 0.1 ms unless FLY_DT is set. The membrane and synapse
     * are integrated exactly, so dt only sets the time resolution of spikes;
     * delay (1.8 ms) and refractory period (2.2 ms) must be whole steps. */
    const double v_0 = -52.0, t_mbr = 20.0, tau = 5.0, w_syn = 0.275;
    const double dt = getenv("FLY_DT") ? atof(getenv("FLY_DT")) : 0.1;
    e->dt = dt;
    if (fabs(lround(1.8 / dt) * dt - 1.8) > 1e-9 || fabs(lround(2.2 / dt) * dt - 2.2) > 1e-9) {
        free(e); return NULL;                         /* delay/refractory not whole steps */
    }
    e->D = (int)lround(1.8 / dt); e->R = (int)lround(2.2 / dt);

    if (n > PE_MAX_N) { free(e); return NULL; }        /* too many neurons to pack */
    e->pe = malloc((size_t)nnz * sizeof(uint32_t));
    e->wlut = malloc((PE_MASK + 1) * sizeof(float));
    for (int q = 0; q < nnz; q++) {
        if (weights[q] < -PE_BIAS || weights[q] >= PE_BIAS) {
            free(e->pe); free(e->wlut); free(e); return NULL;    /* count too large */
        }
        e->pe[q] = ((uint32_t)indices[q] << PE_SHIFT) | (uint32_t)(weights[q] + PE_BIAS);
    }
    for (int c = 0; c <= (int)PE_MASK; c++)
        e->wlut[c] = (float)((int16_t)(c - PE_BIAS) * w_syn);

    double a = 1.0 / t_mbr, b = 1.0 / tau;
    double evd = exp(-a * dt), egd = exp(-b * dt);
    e->ev = (float)evd; e->eg = (float)egd;
    e->kg = (float)(a * (egd - evd) / (a - b));
    e->v0rest = (float)(v_0 * (1.0 - evd));
    e->vth = -45.0f; e->vrst = -52.0f;
    {   /* the float fixed point of the resting update (v*ev + v0rest + 0) */
        float vf = -52.0f, prev;
        int it = 0;
        do { prev = vf; vf = vf * e->ev; vf = vf + e->v0rest; vf = vf + 0.0f; } while (vf != prev && ++it < 100000);
        e->vfix = vf;
    }
    e->qtol = 0.0f; e->x_eps = X_EPS;
    e->poi_w = (float)(w_syn * 250.0);
    e->rng = seed;

    size_t nf = ((size_t)n * 4 + 63) & ~(size_t)63;
    e->v = aligned_alloc(64, nf); e->g = aligned_alloc(64, nf);
    e->rfc_left = malloc(n); e->rfc_len = malloc(n);
    e->silenced = malloc(n); e->is_poi = malloc(n);
    e->quiet = malloc(e->nblocks); e->pinned = malloc(e->nblocks);
    e->spike_counts = malloc((size_t)n * sizeof(int32_t));
    e->spikes_cap = 4096; e->spikes = malloc(e->spikes_cap * sizeof(int32_t));

    if (nthreads < 1) nthreads = 1;
    if (nthreads > MAXT) nthreads = MAXT;
    e->nthreads = nthreads;
    for (int t = 0; t < nthreads; t++) {
        worker *k = &e->wk[t];
        k->e = e; k->tid = t;
        k->lo = (int)(((long)n * t / nthreads) & ~63l);
        k->hi = t == nthreads - 1 ? n : (int)(((long)n * (t + 1) / nthreads) & ~63l);
        k->cap = 1024; k->spk = malloc(k->cap * sizeof(int32_t));
        hash_init(k, 12);                               /* 4096 slots, 48 KB */
        k->ring = calloc(e->D + 1, sizeof(evlist));
        k->rfc_cap = 1024;
        k->rfc = malloc(k->rfc_cap * sizeof(int32_t));
        k->hold_v = malloc(k->rfc_cap * sizeof(float));
        k->hold_g = malloc(k->rfc_cap * sizeof(float));
    }
    /* per-row entry points into each thread's target range (rows sorted) */
    int T1 = nthreads + 1;
    e->split = malloc((size_t)n * T1 * sizeof(int32_t));
    for (int i = 0; i < n; i++) {
        int32_t a = indptr[i], z = indptr[i + 1];
        for (int t = 0; t < nthreads; t++)
            e->split[(long)i * T1 + t] = lower_bound(indices, a, z, e->wk[t].lo);
        e->split[(long)i * T1 + nthreads] = z;
    }
    lif_reset(e);
    /* Cores: the caller takes cpu_base, workers cpu_base+1... Several engines
     * in one machine (parallel data generation) set FLY_CPU_BASE apart. */
    e->ncpu = (int)sysconf(_SC_NPROCESSORS_ONLN) > 0 ? (int)sysconf(_SC_NPROCESSORS_ONLN) : 1;
    e->cpu_base = getenv("FLY_CPU_BASE") ? atoi(getenv("FLY_CPU_BASE")) : 0;
    for (int t = 1; t < nthreads; t++) {
        pthread_create(&e->wk[t].th, NULL, worker_main, &e->wk[t]);
        pin_thread(e->wk[t].th, (e->cpu_base + t) % e->ncpu);
    }
    return e;
}


void lif_destroy(lif *e) {
    atomic_store(&e->quit, 1);
    if (e->has_driver) {
        atomic_fetch_add(&e->job_gen, 1);
        futex_wake(&e->job_gen);
        pthread_join(e->driver, NULL);
    }
    wake_workers(e);
    for (int t = 1; t < e->nthreads; t++) pthread_join(e->wk[t].th, NULL);
    for (int t = 0; t < e->nthreads; t++) {
        worker *k = &e->wk[t];
        for (int s = 0; s <= e->D; s++) { free(k->ring[s].idx); free(k->ring[s].val); free(k->ring[s].val_s); }
        free(k->ring); free(k->spk); hash_free(k);
        free(k->rfc); free(k->hold_v); free(k->hold_g); free(k->xl);
    }
    free(e->pe); free(e->wlut); free(e->pmult); free(e->adapt); free(e->gs); free(e->xflag); free(e->adapt_b); free(e->slow_r); free(e->split); free(e->v); free(e->g); free(e->rfc_left); free(e->rfc_len);
    free(e->silenced); free(e->is_poi); free(e->quiet); free(e->pinned);
    free(e->spike_counts); free(e->spikes);
    free(e->std_f); free(e->std_d); free(e->std_rel); free(e->std_list);
    free(e->poi_idx); free(e->poi_p); free(e->collected);
    free(e);
}

void lif_set_seed(lif *e, uint32_t seed) { e->rng = seed; }

/* Scale every recurrent synapse: w = count * (w_syn * gain). Poisson (sensory)
 * drive is unchanged. For controls that match a network's activity level;
 * gain 1.0 reproduces the published model exactly. (`weights` is unused since
 * weights moved to a lookup table; kept for API compatibility.) */
void lif_set_gain(lif *e, const int16_t *weights, double gain) {
    (void)weights;
    const double w = 0.275 * gain;
    for (int c = 0; c <= (int)PE_MASK; c++)
        e->wlut[c] = (float)((int16_t)(c - PE_BIAS) * w);
}

void lif_set_poisson(lif *e, const int32_t *idx, const double *rates_hz, int m) {
    for (int q = 0; q < e->npoi; q++) {
        e->rfc_len[e->poi_idx[q]] = (uint8_t)e->R;
        e->is_poi[e->poi_idx[q]] = 0;
    }
    memset(e->pinned, 0, e->nblocks);
    e->poi_idx = realloc(e->poi_idx, (size_t)(m ? m : 1) * sizeof(int32_t));
    e->poi_p = realloc(e->poi_p, (size_t)(m ? m : 1) * sizeof(double));
    e->npoi = m;
    for (int q = 0; q < m; q++) {
        double p = rates_hz[q] * e->dt * 1e-3;
        e->poi_p[q] = p < 0 ? 0 : p > 1 ? 1 : p;
        e->poi_idx[q] = idx[q];
        e->rfc_len[idx[q]] = 0; e->rfc_left[idx[q]] = 0; e->is_poi[idx[q]] = 1;
        e->pinned[idx[q] / BLK] = 1; e->quiet[idx[q] / BLK] = 0;
    }
    for (int t = 0; t < e->nthreads; t++) {             /* drop from refractory lists */
        worker *k = &e->wk[t]; int keep = 0;
        for (int q = 0; q < k->nrfc; q++)
            if (e->rfc_left[k->rfc[q]]) k->rfc[keep++] = k->rfc[q];
        k->nrfc = keep;
    }
}

/* Same targets as the last lif_set_poisson, new rates. Equivalent to calling
 * lif_set_poisson again with the same indices (targets are never refractory).
 * Returns -1 if m differs from the current target count. */
int lif_set_poisson_rates(lif *e, const double *rates_hz, int m) {
    if (m != e->npoi) return -1;
    for (int q = 0; q < m; q++) {
        double p = rates_hz[q] * e->dt * 1e-3;
        e->poi_p[q] = p < 0 ? 0 : p > 1 ? 1 : p;
    }
    return 0;
}

void lif_silence(lif *e, const int32_t *idx, int m, int on) {
    for (int q = 0; q < m; q++) e->silenced[idx[q]] = on ? 1 : 0;
}

double lif_dt(lif *e) { return e->dt; }

/* Advance one step (dt, 0.1 ms by default). Returns the number of spikes; lif_spikes() holds
 * their indices in ascending order until the next call. */
int lif_step(lif *e) {
    /* The caller is worker 0; it must not share a core with a spinning worker.
     * Pin it to cpu_base the first time it steps. */
    static __thread int caller_pinned;
    if (!caller_pinned && e->nthreads > 1) {
        pin_thread(pthread_self(), e->cpu_base % e->ncpu);
        caller_pinned = 1;
    }
    /* A. deliver + integrate + threshold non-Poisson neurons (parallel) */
    double t0 = now_us();
    run_phase(e, 0);
    double t1 = now_us();

    /* Poisson drive, serial so the PRNG stream matches, then its threshold */
    float *v = e->v;
    for (int q = 0; q < e->npoi; q++)
        if (mulberry32(&e->rng) < e->poi_p[q]) v[e->poi_idx[q]] += e->poi_w;
    e->nspikes = 0;
    for (int t = 0; t < e->nthreads; t++)             /* ranges ascend with t */
        for (int s = 0; s < e->wk[t].nspk; s++) push_spike(e, e->wk[t].spk[s]);
    int base = e->nspikes;
    for (int q = 0; q < e->npoi; q++)
        if (v[e->poi_idx[q]] > e->vth) push_spike(e, e->poi_idx[q]);
    if (e->nspikes > base) qsort(e->spikes, e->nspikes, sizeof(int32_t), cmp_i32);

    if (e->std_n) {                                   /* short-term depression */
        for (int q = 0; q < e->std_n; q++) { int32_t i = e->std_list[q]; e->std_d[i] = e->std_d[i] * e->std_e; }
        for (int s = 0; s < e->nspikes; s++) {
            int32_t i = e->spikes[s];
            if (e->std_f[i] > 0.0f) {
                float rel = 1.0f - e->std_d[i];
                e->std_rel[i] = rel;
                float t = rel * e->std_f[i];
                e->std_d[i] = 1.0f - t;
            }
        }
    }

    /* B. reset spikers + scatter their output into the delay line (parallel) */
    double t2 = now_us();
    if (e->nspikes) run_phase(e, 1);
    double t3 = now_us();
    e->t_a += t1 - t0; e->t_serial += t2 - t1; e->t_b += t3 - t2;

    e->slot = (e->slot + 1) % (e->D + 1);
    e->step_count++;
    return e->nspikes;
}

int lif_run(lif *e, int steps) {
    int total = 0;
    for (int s = 0; s < steps; s++) total += lif_step(e);
    return total;
}

/* Run `steps` steps and gather every spike, in step order, into one buffer
 * (lif_collected). One foreign call per millisecond instead of ten. */
long lif_run_collect(lif *e, int steps) {
    e->ncollected = 0;
    for (int s = 0; s < steps; s++) {
        int k = lif_step(e);
        if (e->ncollected + k > e->collected_cap) {
            e->collected_cap = (e->ncollected + k) * 2 + 4096;
            e->collected = realloc(e->collected, (size_t)e->collected_cap * sizeof(int32_t));
        }
        memcpy(e->collected + e->ncollected, e->spikes, (size_t)k * sizeof(int32_t));
        e->ncollected += k;
    }
    return e->ncollected;
}

const int32_t *lif_collected(lif *e) { return e->collected; }

/* ---------------------------------------------------------- async driver */
/* lif_start(steps) returns at once; a driver thread (worker 0, pinned to core
 * 0) runs the steps while the caller does other work, e.g. processing the
 * previous block's spikes in Python. lif_wait() blocks until done. The engine
 * must not be touched between the two calls. */
static void *driver_main(void *arg) {
    lif *e = arg;
    int seen = 0;
    for (;;) {
        int gen, spins = 0; double ts = 0;
        while ((gen = atomic_load(&e->job_gen)) == seen) {
            if (keep_spinning(&spins, &ts)) continue;
            atomic_store(&e->driver_sleeping, 1);
            if (atomic_load(&e->job_gen) == seen) futex_wait(&e->job_gen, seen);
            atomic_store(&e->driver_sleeping, 0);
            spins = 0; ts = 0;
        }
        seen = gen;
        if (atomic_load(&e->quit)) return NULL;
        lif_run_collect(e, atomic_load(&e->job_steps));
        atomic_store(&e->job_done, gen);
        if (atomic_load(&e->waiter_sleeping)) futex_wake(&e->job_done);
    }
}

void lif_start(lif *e, int steps) {
    if (!e->has_driver) {
        pthread_create(&e->driver, NULL, driver_main, e);
        e->has_driver = 1;
    }
    atomic_store(&e->job_steps, steps);
    atomic_fetch_add(&e->job_gen, 1);
    if (atomic_load(&e->driver_sleeping)) futex_wake(&e->job_gen);
}

long lif_wait(lif *e) {
    int gen = atomic_load(&e->job_gen), spins = 0, done; double ts = 0;
    while ((done = atomic_load(&e->job_done)) != gen) {
        if (keep_spinning(&spins, &ts)) continue;
        atomic_store(&e->waiter_sleeping, 1);
        if (atomic_load(&e->job_done) == done) futex_wait(&e->job_done, done);
        atomic_store(&e->waiter_sleeping, 0);
        spins = 0; ts = 0;
    }
    return e->ncollected;
}
const int32_t *lif_spikes(lif *e) { return e->spikes; }
float *lif_v(lif *e) { return e->v; }
float *lif_g(lif *e) { return e->g; }
int32_t *lif_spike_counts(lif *e) { return e->spike_counts; }
long lif_step_count(lif *e) { return e->step_count; }
long lif_active_blocks(lif *e) {
    long s = 0;
    for (int t = 0; t < e->nthreads; t++) s += e->wk[t].active_blocks;
    return s;
}
int lif_nblocks(lif *e) { return e->nblocks; }
void lif_phase_times(lif *e, double *out3) {
    out3[0] = e->t_a; out3[1] = e->t_serial; out3[2] = e->t_b;
}
/* Coefficient of a drive decaying with time constant tau_ms in the exact
 * one-step membrane update (t_mbr = 20 ms). */
static double drive_coef(double tau_ms, double dt) {
    const double am = 1.0 / 20.0, c = 1.0 / tau_ms;
    if (fabs(am - c) < 1e-12) return am * dt * exp(-am * dt);
    return am * (exp(-c * dt) - exp(-am * dt)) / (am - c);
}

/* Enable the extended dynamics (see struct lif) with per-neuron values
 * (arrays of n; adapt_mV per neuron, slow_ratio per presynaptic neuron), or
 * disable them when both arrays are all zero, returning to the published model. */
void lif_set_dynamics(lif *e, double tau_adapt_ms, const float *adapt_mV,
                      double tau_slow_ms, const float *slow_ratio) {
    int on = 0;
    for (int i = 0; i < e->n && !on; i++) on = adapt_mV[i] > 0 || slow_ratio[i] > 0;
    if (on) {
        if (!e->adapt) e->adapt = calloc(e->n, sizeof(float));
        if (!e->gs) e->gs = calloc(e->n, sizeof(float));
        if (!e->adapt_b) e->adapt_b = malloc(e->n * sizeof(float));
        if (!e->xflag) e->xflag = calloc(e->n, 1);
        if (!e->slow_r) e->slow_r = malloc(e->n * sizeof(float));
        memcpy(e->adapt_b, adapt_mV, e->n * sizeof(float));
        memcpy(e->slow_r, slow_ratio, e->n * sizeof(float));
        e->ea = (float)exp(-e->dt / tau_adapt_ms); e->ka = (float)drive_coef(tau_adapt_ms, e->dt);
        e->es = (float)exp(-e->dt / tau_slow_ms);  e->ks = (float)drive_coef(tau_slow_ms, e->dt);
    } else {
        free(e->adapt); free(e->gs); free(e->adapt_b); free(e->slow_r); free(e->xflag);
        e->adapt = e->gs = e->adapt_b = e->slow_r = NULL; e->xflag = NULL;
        for (int t = 0; t < e->nthreads; t++) e->wk[t].nxl = 0;
    }
    e->ext = on;
    memset(e->quiet, 0, e->nblocks);
}

/* Short-term depression of output synapses (see struct lif): f[i] in (0, 1]
 * for depressing presynaptic neurons, 0 = none (all 0 disables it). tau_ms =
 * recovery time constant. Depletion starts at 0 (fully recovered). */
void lif_set_std(lif *e, const float *f, double tau_ms) {
    int m = 0;
    for (int i = 0; i < e->n; i++) m += f[i] > 0.0f;
    free(e->std_list); e->std_list = NULL; e->std_n = 0;
    if (!m) { free(e->std_f); free(e->std_d); free(e->std_rel); e->std_f = e->std_d = e->std_rel = NULL; return; }
    if (!e->std_f) { e->std_f = malloc(e->n * sizeof(float)); e->std_d = calloc(e->n, sizeof(float));
                     e->std_rel = malloc(e->n * sizeof(float)); }
    memcpy(e->std_f, f, e->n * sizeof(float));
    e->std_list = malloc((size_t)m * sizeof(int32_t));
    for (int i = 0; i < e->n; i++) if (f[i] > 0.0f) e->std_list[e->std_n++] = i;
    e->std_e = (float)exp(-e->dt / tau_ms);
}
float *lif_std_depletion(lif *e) { return e->std_d; }

float *lif_adapt(lif *e) { return e->adapt; }
float *lif_gs(lif *e) { return e->gs; }

/* Add `vals` to the synaptic drive g of neurons `idx` (between steps only),
 * waking their blocks. For graded, non-spiking inputs modelled outside the
 * spiking network, e.g. the APL neuron's global inhibition of Kenyon cells. */
void lif_add_g(lif *e, const int32_t *idx, const float *vals, int m) {
    for (int q = 0; q < m; q++) {
        e->g[idx[q]] += vals[q];
        e->quiet[idx[q] / BLK] = 0;
    }
}

/* Plasticity: a float multiplier per connection (CSR order, as passed to
 * lif_create), all 1.0 initially. Returns it for the caller to modify between
 * steps; weights are read during steps, so do not write while stepping. */
float *lif_plastic_enable(lif *e) {
    if (!e->pmult) {
        e->pmult = malloc((size_t)e->nnz * sizeof(float));
        for (int q = 0; q < e->nnz; q++) e->pmult[q] = 1.0f;
    }
    return e->pmult;
}

long lif_edges(lif *e) {
    long s = 0;
    for (int t = 0; t < e->nthreads; t++) s += e->wk[t].edges;
    return s;
}
void lif_thread_stats(lif *e, double *out) {   /* per thread: busy_a, busy_b, deliveries, active blocks */
    for (int t = 0; t < e->nthreads; t++) {
        out[4*t] = e->wk[t].busy_a; out[4*t+1] = e->wk[t].busy_b;
        out[4*t+2] = e->wk[t].t_deliv; out[4*t+3] = (double)e->wk[t].active_blocks;
    }
}
