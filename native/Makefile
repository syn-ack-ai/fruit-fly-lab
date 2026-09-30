# Native LIF engine. aarch64 (Raspberry Pi 5, Apple Silicon) uses NEON
# intrinsics; other targets use a plain-C path the compiler vectorises.
# -ffp-contract=off is required everywhere: fused multiply-adds would change
# float32 rounding and break bit-exactness with the Python and browser engines.
ARCH := $(shell uname -m)
OS := $(shell uname -s)
ifeq ($(OS),Darwin)
  TUNE =
else ifeq ($(ARCH),aarch64)
  TUNE = -mcpu=cortex-a76
else
  TUNE = -march=native
endif
CFLAGS = -O3 -g $(TUNE) -ffp-contract=off -fno-fast-math -Wall -Wextra -fPIC

liblif.so: lif_native.c
	$(CC) $(CFLAGS) -shared -o $@.tmp $< -lm -lpthread && mv -f $@.tmp $@   # atomic: never overwrite a loaded library in place

generic: lif_native.c
	$(CC) $(CFLAGS) -DLIF_GENERIC -shared -o liblif_generic.so $< -lm -lpthread

# CUDA engine (same C API; FLY_NATIVE_LIB=native/liblif_cuda.so). nvcc from
# $$CUDA_HOME, else a user-level conda env: conda create -n cuda -c conda-forge
# cuda-nvcc=12.8 cuda-cudart-dev=12.8 cuda-cccl=12.8. -fmad=false keeps float32
# results bit-identical with the CPU engine; -ffp-contract=off does the same for
# the host code (aarch64 g++ fuses by default). -arch=native = the local GPU.
CUDA_HOME ?= $(firstword $(wildcard $(HOME)/miniforge3/envs/cuda /usr/local/cuda))
NVCC ?= $(CUDA_HOME)/bin/nvcc
# libraries: lib in a conda env, lib64 in a system install (/usr/local/cuda, e.g. JetPack)
CUDA_LIB ?= $(firstword $(wildcard $(CUDA_HOME)/lib64 $(CUDA_HOME)/lib))
# host compiler nvcc accepts (CUDA 12.8: gcc <= 14; conda-forge gxx_linux-64=14 in the same env)
CCBIN ?= $(firstword $(wildcard $(CUDA_HOME)/bin/x86_64-conda-linux-gnu-g++ $(CUDA_HOME)/bin/aarch64-conda-linux-gnu-g++) g++)
cuda: lif_cuda.cu
	$(NVCC) -ccbin $(CCBIN) -O3 -arch=native -fmad=false -Xcompiler -fPIC,-ffp-contract=off -shared -o liblif_cuda.so.tmp $< \
	  -L$(CUDA_LIB) -Xlinker -rpath=$(CUDA_LIB) -lcudart -lpthread && mv -f liblif_cuda.so.tmp liblif_cuda.so

clean:
	rm -f liblif.so liblif_generic.so liblif_cuda.so
.PHONY: clean generic cuda
