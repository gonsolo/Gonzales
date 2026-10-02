#!/usr/bin/env python3
"""Research tool: which registers does the trace instruction write? Replaces the kernel's result stores with stores of
chosen registers (six per run), traces a TLAS over three BLASes (as_probe) and prints, per register, its values for
rays that hit mesh 0 / 1 / 2 (Vulkan's own mesh id), so a register that carries the instance index shows up.
Usage: dump_regs.py [first_reg last_reg]"""
import os, re, subprocess, sys, tempfile
here = os.path.dirname(os.path.abspath(__file__)); root = os.path.abspath(f"{here}/../../..")
lo, hi = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 47)
d = tempfile.mkdtemp(prefix="dumpregs_")
def run(*a, **k): return subprocess.run(a, check=True, capture_output=True, text=True, **k)
run("glslc", "--target-env=vulkan1.2", "-O0", f"{root}/src/vulkanrt/shaders/intersect_batch.comp", "-o", f"{d}/s.spv")
run("gcc", "-O1", f"{root}/docs/rtcore/pipeline_dump.c", "-lvulkan", "-o", f"{d}/pipeline_dump")
txt = run(f"{d}/pipeline_dump", d, f"{d}/s.spv").stdout
size = int(re.search(r"Binary Size = (\d+)", txt).group(1))
blob = subprocess.run(["zstd", "-d", "-q", "-c"], input=open(f"{d}/pipeline_cache.bin", "rb").read()[0x64:], capture_output=True, check=True).stdout
start = next(o for o in range(0x100, len(blob) - 16, 8) if blob[o] == 0x19 and blob[o + 1] == 0x79 and blob[o + 9] == 0x25)
open(f"{d}/code.bin", "wb").write(blob[start:start + size])
run("gcc", "-O1", f"{here}/as_probe.c", f"-I{root}/src/vulkanrt", f"-L{root}/build", "-lvulkanrt", "-lm", f"-Wl,-rpath,{root}/build", "-o", f"{d}/as_probe")
run("gcc", "-O1", f"{here}/run_rt4.c", "-I/opt/cuda/include", "-L/opt/cuda/lib64", "-lcuda", "-o", f"{d}/run_rt4")
run(f"{d}/as_probe", f"{d}/probe", "4", "1024", "3")
ref = [l.split() for l in open(f"{d}/probe/ref.txt")]
mesh = [int(r[6]) if int(r[1]) else -1 for r in ref]
os.makedirs("/tmp/ncu", exist_ok=True)
import struct
cols = {}
regs = [r for r in range(lo, hi + 1) if r not in (10, 11, 12, 13, 14, 26)]
for g in range(0, len(regs), 6):
    grp = regs[g:g + 6]
    while len(grp) < 6: grp.append(grp[-1])
    sasm = open(f"{here}/trace_kernel.sasm").read().split(".at 0x07f0")[0] + ".at 0x07f0\n    ULDC.64 UR4, c[0x0][0x50]\n"
    for k, off in enumerate((4, 8, 12, 16, 20)): sasm += f"    IADD3 R{10 + k}, R26, 0x{off:x}, RZ\n"
    addr = ["R26", "R10", "R11", "R12", "R13", "R14"]
    for a, r in zip(addr, grp): sasm += f"    STG.E [{a}.U32+UR4], R{r}\n"
    sasm += "    EXIT\n"
    open(f"{d}/dump.sasm", "w").write(sasm)
    env = dict(os.environ, TRACE_SASM=f"{d}/dump.sasm")
    run("python3", f"{here}/make_trace_kernel.py", f"{d}/code.bin", f"{d}/code_dump.bin", env=env)
    run("python3", f"{here}/make_cubin.py", f"{d}/code_dump.bin", f"{d}/dump.cubin", "56")
    out = subprocess.run([f"{d}/run_rt4", f"{d}/dump.cubin", f"{d}/probe", "3"], capture_output=True, text=True)
    if "OK" not in out.stdout: print("launch failed for", grp, out.stdout, out.stderr); continue
    res = open("/tmp/ncu/res.bin", "rb").read()
    for i, r in enumerate(grp[:6]):
        cols[r] = [struct.unpack_from("<I", res, n * 32 + 4 * i)[0] for n in range(len(mesh))]
for r in sorted(cols):
    by = {m: sorted(set(cols[r][n] for n in range(len(mesh)) if mesh[n] == m)) for m in (-1, 0, 1, 2)}
    interesting = any(len(v) > 0 for v in by.values())
    desc = "  ".join(f"mesh{m}:{[hex(x) for x in v[:3]]}{'+' if len(v) > 3 else ''}" for m, v in by.items())
    print(f"R{r:<2} {desc}")
