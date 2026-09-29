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

## How it differs inside

- Every neuron is integrated every step (the CPU's quiescent-block skipping is
  an optimisation that never changes results).
- Synaptic input per step is summed in float64 with atomics into a dense
  (D+1) x n delay ring and cast to float32 at delivery, like the CPU's float64
  hash: exact in any order for the model's weights.
- Poisson draws reproduce the CPU's mulberry32 stream by index.
- Per-step counters live on the GPU; each run of N steps (N <= 100) is captured
  once as a CUDA graph and replayed.
- v, g, adapt, gs are mirrored to the host when Python reads them
  (`NativeLIFEngine.v` etc. call `lif_sync_host`) and uploaded by `wake_all()`;
  spike counts are kept on the host from the collected spikes.
- Plasticity multipliers live in a host array: `plastic_multipliers()` marks
  them for a full upload before the next run, and `commit_plastic(positions)`
  uploads a changed subset (MushroomBody does this after each update).
  With multipliers that push weights far below 2^-25 mV, float64 sums could
  differ from the CPU's order in the last bit.

## Speed (RTX 3080 Ti vs i9-9900K, 8 threads)

| Workload | CPU | CUDA |
| --- | --- | --- |
| whole brain, resting ORN input | 0.4-1.0x real time | 4.5-5.4x |
| whole brain, looming | 0.8-1.5x | 4.3-5.0x |
| closed-loop world (fly.world.run) | 0.24x | 0.61x (Python-bound) |
| Habitat follow-a-person brain client | 4.6 ms / sim ms | 1.5 ms / sim ms |

The GPU cost is ~20 us per 0.1 ms step, set by five small kernels per step, not
by the number of neurons or spikes.
