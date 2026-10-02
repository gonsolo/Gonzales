#!/usr/bin/env python3
"""Builds build/rt_trace.cubin: the kernel rtcore.cpp loads. Needs a Vulkan device with ray query, glslc, zstd,
nvcc, nvdisasm. The NVIDIA driver compiles src/vulkanrt/shaders/intersect_batch.comp; the machine code is read from
the pipeline cache, patched (make_trace_kernel.py) and wrapped in a cubin (make_cubin.py). Nothing from NVIDIA is
committed: the cubin is generated on the machine that uses it.
Usage: build_rt_cubin.py [out.cubin]"""
import os, re, subprocess, sys, tempfile
here = os.path.dirname(os.path.abspath(__file__)); root = os.path.abspath(f"{here}/../../..")
out = sys.argv[1] if len(sys.argv) > 1 else f"{root}/build/rt_trace.cubin"
d = tempfile.mkdtemp(prefix="rtcubin_")
def run(*a, **k): return subprocess.run(a, check=True, capture_output=True, text=True, **k)
run("glslc", "--target-env=vulkan1.2", "-O0", f"{root}/src/vulkanrt/shaders/intersect_batch.comp", "-o", f"{d}/s.spv")
run("gcc", "-O1", f"{root}/docs/rtcore/pipeline_dump.c", "-lvulkan", "-o", f"{d}/pipeline_dump")
txt = run(f"{d}/pipeline_dump", d, f"{d}/s.spv").stdout
size = int(re.search(r"Binary Size = (\d+)", txt).group(1))
cache = open(f"{d}/pipeline_cache.bin", "rb").read()
open(f"{d}/c.zst", "wb").write(cache[0x64:]); run("zstd", "-d", "-f", "-q", f"{d}/c.zst", "-o", f"{d}/c.raw")
blob = open(f"{d}/c.raw", "rb").read()
# machine code starts with S2R Rn, SR_CTAID.X (opcode 0x919, special register 0x25 in byte 9)
start = next(o for o in range(0x100, len(blob) - 16, 8) if blob[o] == 0x19 and blob[o + 1] == 0x79 and blob[o + 9] == 0x25)
open(f"{d}/code.bin", "wb").write(blob[start:start + size])
run("python3", f"{here}/make_trace_kernel.py", f"{d}/code.bin", f"{d}/code_trace.bin")
os.makedirs(os.path.dirname(out), exist_ok=True)
print(run("python3", f"{here}/make_cubin.py", f"{d}/code_trace.bin", out, "56").stdout.strip())
conv = os.path.join(os.path.dirname(out), "rt_convert.cubin")
run("nvcc", "-cubin", "-arch=sm_86", "-O2", f"{root}/src/rtcore/rt_convert.cu", "-o", conv)
print("wrote", conv)
