#!/usr/bin/env python3
"""Extract the NVIDIA driver's compiled SASS for a Vulkan compute pipeline and disassemble it with nvdisasm.

Usage: pipeline_dump <outdir>   (writes pipeline_cache.bin; prints "Binary Size" in bytes)
       extract_sass.py <outdir> <binary_size> [code_offset=0x178]

The pipeline cache from vkGetPipelineCacheData holds a zstd frame (magic 28 b5 2f fd) at byte 0x64. The
decompressed blob starts with "NVDANVVMNVuc"; for our compute shader the machine code starts at 0x178 and is
`Binary Size` bytes long. nvdisasm needs a cubin, so the bytes are patched into the .text of a dummy cubin.
nvdisasm silently OMITS instructions it does not know; list them with the gap finder below.
"""
import re, struct, subprocess, sys, os
out = sys.argv[1]; size = int(sys.argv[2]); start = int(sys.argv[3], 0) if len(sys.argv) > 3 else 0x178
raw_cache = open(f"{out}/pipeline_cache.bin", "rb").read()
open(f"{out}/cache.zst", "wb").write(raw_cache[0x64:])
subprocess.run(["zstd", "-d", "-f", f"{out}/cache.zst", "-o", f"{out}/cache.raw"], check=True, capture_output=True)
blob = open(f"{out}/cache.raw", "rb").read()
code = blob[start:start + size]
open(f"{out}/dummy.cu", "w").write('extern "C" __global__ void k(float* a,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;float x=a[i];\n'
    '#pragma unroll 1\nfor(int j=0;j<n;j++){x=x*1.0001f+sinf(x)+cosf(x*2.f)+expf(x)+logf(x+3.f)+sqrtf(x+4.f)+tanf(x)+a[(i+j)%n];}a[i]=x;}')
subprocess.run(["nvcc", "-cubin", "-arch=sm_86", "-O0", f"{out}/dummy.cu", "-o", f"{out}/dummy.cubin"], check=True)
cub = bytearray(open(f"{out}/dummy.cubin", "rb").read())
shoff = struct.unpack_from("<Q", cub, 0x28)[0]; es, sn, sx = struct.unpack_from("<HHH", cub, 0x3A)
stro = struct.unpack_from("<Q", cub, shoff + sx * es + 0x18)[0]
for i in range(sn):
    nm, _, _, _, off, sz = struct.unpack_from("<IIQQQQ", cub, shoff + i * es)
    if cub[stro + nm:cub.index(b"\0", stro + nm)] == b".text.k":
        assert sz >= size, "dummy kernel too small"
        cub[off:off + size] = code
open(f"{out}/patched.cubin", "wb").write(cub)
sass = subprocess.run(["nvdisasm", "-c", f"{out}/patched.cubin"], capture_output=True, text=True).stdout
open(f"{out}/shader.sass", "w").write(sass)
seen = {int(m.group(1), 16) for m in re.finditer(r"/\*([0-9a-f]{4})\*/\s+\S", sass)}
print("instructions:", size // 16, "decoded:", len(seen))
for o in range(0, size, 16):
    if o not in seen:
        ins = code[o:o + 16]
        print(f"UNDECODED {o:04x} opcode={(ins[0] | (ins[1] << 8)) & 0xfff:#05x}", ins.hex(" ", 4))
