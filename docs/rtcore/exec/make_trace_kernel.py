#!/usr/bin/env python3
"""Turn the driver's compiled ray-query shader into a kernel that only runs the hardware trace and stores the raw
result registers. Steps (offsets are into the 5376-byte shader from intersect_batch.comp):
  1. 0x0600  LOP3 R21, R25, 0x55000   ->  IMAD.MOV.U32 R21, RZ, RZ, RZ   (R21 = 0: see NOTES.md)
  2. from 0x7f0 (after the trace instruction 0x9d4 at 0x6e0 jumped to 0x7d0), replace the driver's post-processing
     with: ULDC.64 UR4,results ; three IADD3 for the field addresses ; STG R21,R22,R23,R20 ; EXIT.
The instructions are in trace_kernel.sasm (pseudo SASS); this script assembles and verifies them.
Output record per ray (24 bytes used of 32): [t, u, v, raw R20, raw R8, raw R9]."""
import os, sys, struct
src, dst = sys.argv[1], sys.argv[2]
c = bytearray(open(src, "rb").read())
# The offsets below are specific to the code NVIDIA's compiler produced for intersect_batch.comp (driver 615.71.09).
# Refuse to patch anything that does not look the same.
import re, subprocess
_txt = subprocess.run(["nvdisasm", "--binary", "SM86", src], capture_output=True, text=True).stdout
def _at(off): m = re.search(r"/\*%04x\*/\s+(.*?)\s*;" % off, _txt); return m.group(1) if m else None
_expect = {0x600: "LOP3.LUT R21, R25, 0x55000, RZ, 0xfc, !PT", 0x2d0: "IMAD.MOV.U32 R28, RZ, RZ, RZ",
           0x1260: "ULDC.64 UR4, c[0x0][0x50]", 0x1230: "IADD3 R10, R26.reuse, 0xc, RZ", 0x12e0: "@P1 STG.E [R26.U32+UR4], R5",
           0x40: "@P0 EXIT"}
for _o, _e in _expect.items():
    if _at(_o) != _e: sys.exit(f"unexpected code at {_o:#x}: {_at(_o)!r} (wanted {_e!r}): the driver compiled the shader differently")
if (c[0x6e1] << 8 | c[0x6e0]) & 0xfff != 0x9d4: sys.exit("trace instruction not at 0x6e0")
def ins(off): return bytearray(c[off:off + 16])
# Templates: copies of instructions the driver already emitted, with the register/immediate fields patched.
T_MOV, T_ULDC, T_IADD, T_STG, T_EXIT = ins(0x2d0), ins(0x1260), ins(0x1230), ins(0x12e0), ins(0x40)
for t in (T_STG, T_EXIT): t[1] = (t[1] & 0x0f) | 0x70                      # predicate: always
def reg(s): return 255 if s == "RZ" else int(s[1:])

def assemble(line):
    """One pseudo-SASS line -> 16 bytes. Only the forms listed in trace_kernel.sasm exist."""
    if m := re.fullmatch(r"IMAD\.MOV\.U32 (R\d+), RZ, RZ, RZ", line):
        i = bytearray(T_MOV); i[2] = reg(m[1]); return i
    if line == "ULDC.64 UR4, c[0x0][0x50]": return bytearray(T_ULDC)
    if m := re.fullmatch(r"IADD3 (R\d+), R26, (0x[0-9a-f]+), RZ", line):
        i = bytearray(T_IADD); i[2] = reg(m[1]); i[4] = int(m[2], 16); return i
    if m := re.fullmatch(r"STG\.E \[(R\d+)\.U32\+UR4\], (R\d+)", line):
        i = bytearray(T_STG); i[3] = reg(m[1]); i[4] = reg(m[2]); return i
    if line == "EXIT": return bytearray(T_EXIT)
    sys.exit(f"cannot assemble: {line!r}")

listing, at = [], None                                                      # [(offset, text)]
for raw in open(os.environ.get("TRACE_SASM") or __file__.replace("make_trace_kernel.py", "trace_kernel.sasm")):
    line = raw.split("#")[0].strip()
    if not line: continue
    if line.startswith(".at"): at = int(line.split()[1], 16); continue
    listing.append((at, line)); c[at:at + 16] = assemble(line); at += 16
open(dst, "wb").write(c)

# Self-check: disassemble what was written and compare with the listing (modulo spacing and the reuse hints).
_out = subprocess.run(["nvdisasm", "--binary", "SM86", dst], capture_output=True, text=True).stdout
_norm = lambda t: re.sub(r"\.reuse|\s+", "", t.replace("!PT", ""))
for off, text in listing:
    got = re.search(r"/\*%04x\*/\s+(?:@\S+\s+)?(.*?)\s*;" % off, _out)
    if not got or _norm(got[1]) != _norm(text): sys.exit(f"mismatch at {off:#x}: listing {text!r}, nvdisasm {got and got[1]!r}")
print("wrote", dst, "-", len(listing), "instructions assembled and verified")
