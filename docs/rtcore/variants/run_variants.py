#!/usr/bin/env python3
"""Compile each *.comp to SPIR-V, have the driver compile it (pipeline_dump), pull the SASS out of the pipeline
cache and list the instructions nvdisasm cannot decode. Prints the undecoded block per variant."""
import glob, os, re, subprocess, sys
here = os.path.dirname(os.path.abspath(__file__)); out = "/tmp/ncu/variants"; os.makedirs(out, exist_ok=True)
res = {}
for comp in sorted(glob.glob(f"{here}/*.comp")):
    name = os.path.basename(comp)[:-5]; d = f"{out}/{name}"; os.makedirs(d, exist_ok=True)
    subprocess.run(["glslc", "--target-env=vulkan1.2", "-O0", comp, "-o", f"{d}/s.spv"], check=True)
    txt = subprocess.run(["/tmp/pipeline_dump", d, f"{d}/s.spv"], capture_output=True, text=True).stdout
    size = int(re.search(r"Binary Size = (\d+)", txt).group(1))
    cache = open(f"{d}/pipeline_cache.bin", "rb").read()
    open(f"{d}/c.zst", "wb").write(cache[0x64:])
    subprocess.run(["zstd", "-d", "-f", f"{d}/c.zst", "-o", f"{d}/c.raw"], check=True, capture_output=True)
    blob = open(f"{d}/c.raw", "rb").read()
    start = next(o for o in range(0x100, len(blob) - 16, 8) if blob[o] == 0x19 and blob[o + 1] == 0x79 and blob[o + 9] == 0x25)
    code = blob[start:start + size]; open(f"{d}/code.bin", "wb").write(code)
    sass = subprocess.run(["nvdisasm", "--binary", "SM86", f"{d}/code.bin"], capture_output=True, text=True).stdout
    open(f"{d}/code.sass", "w").write(sass)
    seen = {int(m.group(1), 16) for m in re.finditer(r"/\*([0-9a-f]{4})\*/\s+\S", sass)}
    und = [(o, code[o:o + 16]) for o in range(0, size, 16) if o not in seen]
    res[name] = (size, start, und)
    print(f"== {name}: size={size} start={start:#x} undecoded={len(und)}")
    for o, ins in und:
        print(f"   {o:04x} op={(ins[0] | (ins[1] << 8)) & 0xfff:#05x} {ins.hex(' ', 4)}")
