# CUDA brain engine

`native/lif_cuda.cu` builds `native/liblif_cuda.so`, a GPU version of the native
engine with the same C API. Select it with

    FLY_NATIVE_LIB=$PWD/native/liblif_cuda.so

(the CPU `liblif.so` stays the default everywhere). Results are bit-identical
with the CPU engine: `python native/verify_cuda.py` compares both on published
and calibrated dynamics, silencing, input switches, gain, the pipelined
start/wait path, the quiescence tolerance and FLY_DT=0.2. Short-term
depression decays lazily in both (brought up to date when a neuron spikes),
with the same binary-powering decay (`std_decay`): libm's and CUDA's `exp`
may round differently (review 2026-09-28; verified identical on FAFB and the
merged brain, whose depression is brain-wide).

## Setup (user level, no sudo)

    ~/miniforge3/bin/conda create -n cuda -c conda-forge \
        cuda-nvcc=12.8 cuda-cudart-dev=12.8 cuda-cccl=12.8 "gxx_linux-64=14"
    make -C native cuda

The Makefile finds nvcc in `$CUDA_HOME` (default `~/miniforge3/envs/cuda`, then
`/usr/local/cuda`) and a host compiler nvcc accepts (CUDA 12.8 needs gcc <= 14;
the conda `gxx_linux-64=14` is used when the system gcc is newer).
`-arch=native` targets the local GPU (sm_86 on the RTX 3080 Ti; sm_87 on a
Jetson Orin, where JetPack's CUDA in /usr/local/cuda is used).

On a Jetson (JetPack 6/7), CUDA comes with `sudo apt install nvidia-jetpack`;
`make -C native cuda` then uses /usr/local/cuda (libraries in lib64). For
benchmarks, `sudo jetson_clocks` pins the CPU, GPU and memory (EMC) clocks at
their maximum for the power mode (it is not kept across a reboot).

## How it works (rewritten 2026-09-29 for the Jetson Orin Nano)

A run of up to 1000 steps is one cooperative kernel launch, `k_run`, with one
grid barrier per 0.1 ms step (the old engine launched five small kernels per
step, which on the Orin's 8-SM GPU cost more than the work). Launches are
captured in CUDA graphs, cached by their exact arguments.

- **Update (U).** Each warp owns 32-neuron words, dealt round-robin over the
  warps (balanced load); a neuron's state is only touched by its warp. A
  spiking neuron is reset at once (v, g, adaptation, refractory period and
  short-term depression: what the CPU does after its update) and appended to
  the step's spike list. Per-neuron flags (refractory steps, Poisson-driven,
  has adaptation state) are one byte; arrays are padded to whole words.
- **Delay ring.** Synaptic input is summed in float64 with atomics into a
  dense (D+1) x n ring and cast to float32 at delivery, like the CPU's float64
  hash: exact in any order for the model's weights. A bitmap per ring slot
  marks the neurons with input pending, so a step reads the ring only there.
- **Scatter (S).** After the barrier the step's spikes are scattered, each
  spike's synapses split over up to 16 warps. S(k) needs no barrier before
  U(k+1): it writes the ring slot read D steps later.
- **Order.** The GPU lists a step's spikes in any order; the host sorts each
  step's spikes (insertion sort, or a bitmap for large steps), which is the
  CPU's ascending order.
- **Poisson.** Draws reproduce the CPU's mulberry32 stream by index; the CPU's
  `u / 2^32 < p` is the exact integer test `u <= ceil(p 2^32) - 1`.
- **Memory.** When they fit (5 bytes a neuron), each block keeps its neurons'
  v and flags in shared memory for the whole launch; v, g, adaptation and slow
  inhibition are kept in L2 with a persisting access-policy window where the
  GPU has one. The spike buffer (integrated GPU), per-step starts and counts
  are mapped host memory, so nothing is copied back after a run.
- Every neuron is integrated every step (the CPU's quiescent-block skipping is
  an optimisation that never changes results); the approximate quiescence
  tolerance is reproduced per 16-neuron block (a separate kernel variant).
- v, g, adapt, gs are mirrored to the host when Python reads them
  (`NativeLIFEngine.v` etc. call `lif_sync_host`) and uploaded by `wake_all()`;
  spike counts are kept on the host from the collected spikes.
- Plasticity multipliers live in a host array: `plastic_multipliers()` marks
  them for a full upload before the next run, and `commit_plastic(positions)`
  uploads a changed subset (MushroomBody does this after each update).
  With multipliers that push weights far below 2^-25 mV, float64 sums could
  differ from the CPU's order in the last bit.
- FLY_DT must be >= 0.035 ms (the refractory counter has 6 bits).

## Speed

The complete (merged) brain, 165,122 neurons, calibrated dynamics, resting
olfactory input, run in 1 ms blocks as the robot does (in brackets: one 2 s
run). Multiples of real time:

| Machine | step | CPU engine | old CUDA engine | CUDA engine |
| --- | --- | --- | --- | --- |
| Jetson Orin Nano Super (MAXN_SUPER, jetson_clocks) | 0.1 ms | 1.02x (6 threads) | 0.42x (0.44x) | **2.1x** (2.6x) |
| | 0.2 ms | 1.76x | 0.79x | **3.5x** (4.8x) |
| RTX 3080 Ti / i9-9900K | 0.1 ms | | 3.7x (3.6x) | **11.9x** (17.8x) |
| | 0.2 ms | | | **17.9x** (31x) |

On the Orin the step now costs ~39 us: ~30 us of update, ~4 us of scatter and
a ~3 us barrier. Nsight Compute (2026-09-30; `FLY_CUDA_NO_GRAPH=1` launches
directly so the profiler sees the kernel) shows the update issue-bound: ~170
warp instructions per 32 neurons, only ~11% of them floating-point (the rest
branch bookkeeping, moves and integer work for the per-neuron cases:
adaptation, Poisson input, refractory periods, pending input), issue slots 66%
busy with ~2 eligible warps per scheduler. What did not shorten the step:
balancing the blocks (1024-thread blocks, or warps taking words from a
counter), prefetching the flag-dependent loads a word ahead, branch-free
adaptation, integer instead of boolean flags, and v, g, adapt, gs as one
16-byte record (the last made published dynamics 20% slower: more L2 misses). A 1 ms block adds ~80 us
(launch ~13 us, GPU start ~40 us, Python ~30 us). A whole-brain storm (every
neuron driven at 150-600 Hz, 59 M spikes) takes 3.2 s on the Orin's GPU and
31 s on its CPU.
