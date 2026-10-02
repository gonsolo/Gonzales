#!/usr/bin/env python3
"""Turn the driver's compiled ray-query shader into a kernel that only runs the hardware trace and stores the raw
result registers. Steps (offsets are into the 5376-byte shader from intersect_batch.comp):
  1. 0x0600  LOP3 R21, R25, 0x55000   ->  IMAD.MOV.U32 R21, RZ, RZ, RZ   (R21 = 0: see NOTES.md)
  2. from 0x7f0 (after the trace instruction 0x9d4 at 0x6e0 jumped to 0x7d0), replace the driver's post-processing
     with: ULDC.64 UR4,results ; three IADD3 for the field addresses ; STG R21,R22,R23,R20 ; EXIT.
Output record per ray (16 bytes used of 32): [t, u, v, raw R20]."""
import sys, struct
src, dst = sys.argv[1], sys.argv[2]
c = bytearray(open(src, "rb").read())
def ins(off): return bytearray(c[off:off + 16])
mov = ins(0x2d0); mov[2] = 0x15; c[0x600:0x610] = mov                     # IMAD.MOV.U32 R21, RZ, RZ, RZ
uldc = ins(0x1260)                                                         # ULDC.64 UR4, c[0x0][0x50]
iadd = ins(0x1230)                                                         # IADD3 R10, R26, 0xc, RZ
stg = ins(0x12e0); stg[1] = (stg[1] & 0x0f) | 0x70                         # STG.E [R26.U32+UR4], R5  (always)
ext = ins(0x40);   ext[1] = (ext[1] & 0x0f) | 0x70                         # EXIT (always)
def add(dest, imm): i = bytearray(iadd); i[2] = dest; i[4] = imm; return i
def store(addr_reg, data_reg): i = bytearray(stg); i[3] = addr_reg; i[4] = data_reg; return i
seq = [uldc, add(10, 4), add(11, 8), add(12, 12), store(26, 21), store(10, 22), store(11, 23), store(12, 20), ext]
o = 0x7f0
for i in seq: c[o:o + 16] = i; o += 16
open(dst, "wb").write(c)
print("wrote", dst)
