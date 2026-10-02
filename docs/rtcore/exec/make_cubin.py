#!/usr/bin/env python3
"""Build a loadable sm_86 cubin whose kernel `k` has the driver-compiled ray-query code as its .text.
Usage: make_cubin.py <code.bin> <out.cubin> [regs=56]
The skeleton comes from a dummy CUDA kernel; .text.k is overwritten (it must be at least as large) and the
register count in .nv.info and in the section header is raised to `regs`."""
import struct, subprocess, sys, os, tempfile
code = open(sys.argv[1], "rb").read(); out = sys.argv[2]; regs = int(sys.argv[3]) if len(sys.argv) > 3 else 56
d = tempfile.mkdtemp()
open(f"{d}/d.cu", "w").write("""extern "C" __global__ void k(unsigned int rc, unsigned long long as, unsigned long long rays, unsigned int rb, unsigned long long res, unsigned int sb, unsigned long long c10, unsigned long long c18){
 unsigned long long* a=(unsigned long long*)res; unsigned long long b=as, c=rays, e=c10+c18+rb+sb; int n=rc; int i=blockIdx.x*blockDim.x+threadIdx.x; float v[72];
 #pragma unroll
 for(int j=0;j<72;j++) v[j]=(float)a[(i+j)%n]+(float)b*j;
 #pragma unroll 1
 for(int t=0;t<n;t++){
  #pragma unroll
  for(int j=0;j<72;j++){ v[j]=v[j]*1.0001f+sinf(v[(j+1)%72])+cosf(v[(j+5)%72])+expf(v[(j+9)%72])+(float)c+(float)e; }
 }
 volatile unsigned char scr[LMEM]; scr[i&(LMEM-1)]=(unsigned char)n; float s=0;
 #pragma unroll
 for(int j=0;j<72;j++) s+=v[j]; s+=scr[(i*7)&(LMEM-1)];
 a[i]=(unsigned long long)s; }""")
subprocess.run(["nvcc", "-cubin", "-arch=sm_86", "-O0", f"-DLMEM={os.environ.get('LMEM', '4096')}", f"{d}/d.cu", "-o", f"{d}/d.cubin"], check=True)
cub = bytearray(open(f"{d}/d.cubin", "rb").read())
shoff = struct.unpack_from("<Q", cub, 0x28)[0]; es, sn, sx = struct.unpack_from("<HHH", cub, 0x3A)
stro = struct.unpack_from("<Q", cub, shoff + sx * es + 0x18)[0]
secs = {}
for i in range(sn):
    o = shoff + i * es; nm, ty, fl, ad, off, sz, link, info = struct.unpack_from("<IIQQQQII", cub, o)
    secs[cub[stro + nm:cub.index(b"\0", stro + nm)].decode()] = (o, off, sz)
o, off, sz = secs[".text.k"]
assert sz >= len(code), f"dummy text too small: {sz} < {len(code)}"
code = bytearray(code)
if os.environ.get("PATCH_CBANK", "1") == "1":
    import re
    MAP = {(0, 0x20): 0x160, (0, 0x30): 0x168, (0, 0x34): 0x16c, (0, 0x40): 0x170, (0, 0x48): 0x178,
           (0, 0x50): 0x180, (0, 0x58): 0x188, (1, 0x0): 0x190, (1, 0x8): 0x198}
    txt = subprocess.run(["nvdisasm", "--binary", "SM86", sys.argv[1]], capture_output=True, text=True).stdout
    n = 0
    for m in re.finditer(r"/\*([0-9a-f]{4})\*/[^;]*?c\[0x([0-9a-f]+)\]\[0x([0-9a-f]+)\]", txt):
        o = int(m.group(1), 16); key = (int(m.group(2), 16), int(m.group(3), 16))
        dw = struct.unpack_from("<I", code, o + 4)[0]
        dw = (dw & ~((0xffff << 6) | (0x1f << 22))) | (MAP[key] << 6)
        struct.pack_into("<I", code, o + 4, dw); n += 1
    print("patched constant-bank operands:", n)
cub[off:off + len(code)] = code
# section header sh_info: register count in the top byte
info_off = o + 0x2C; info = struct.unpack_from("<I", cub, info_off)[0]
struct.pack_into("<I", cub, info_off, (info & 0x00ffffff) | (regs << 24))
# EIATTR_REGCOUNT (fmt 0x04, attr 0x2f, size 8): <symidx><regs>
o2, off2, sz2 = secs[".nv.info.k"]
p = off2
while p < off2 + sz2:
    fmt, attr = cub[p], cub[p + 1]; size = struct.unpack_from("<H", cub, p + 2)[0]
    if fmt == 4 and attr == 0x2f: struct.pack_into("<I", cub, p + 8, regs)
    p += 4 + (size if fmt == 4 else 0)
open(out, "wb").write(cub)
print("wrote", out, "text", sz, "code", len(code), "regs", regs)
