#!/usr/bin/env python3
"""Research kernel for the AABB candidate protocol (NOTES.md, "Custom primitives"): the driver's intersect_batch code with the
trace re-issued from the returned state until it is empty (P0 = P1 = true resume). Record k of ray i lands at out + 512 i + 32 k:
R20, R21, R22, R23, R8, R9, R10, R11. Usage: SASM_CODE=code.bin aabb_loop.py code_trace.bin out_code.bin
(code_trace.bin = make_trace_kernel.py output; then make_cubin.py, and a runner with a 512-byte record per ray)."""
import sys
from sasm import *
c = bytearray(open(sys.argv[1], 'rb').read())
patch(c, 0x590, [movi(47, 0)])                                         # record offset
k = [shl(30, 26, 0x10), iaddr(30, 30, 47)] + [iaddi(31 + j, 30, 4 * (j + 1)) for j in range(7)]
k += [stg(30 + j, r) for j, r in enumerate((20, 21, 22, 23, 8, 9, 10, 11))]
k += [iaddi(47, 47, 0x20), iaddr(36, 8, 9), isetp_ne(1, 36, 255), isetp_ne(0, 36, 255), movi(37, 0x200), isetp_ne(2, 47, 37)]
k += [movi(r, v) for r, v in FLAGS] + [exit_(pred=2, neg=1)]
k += [bra(0x800 + 16 * len(k), 0x6d0, pred=1), exit_()]
patch(c, 0x800, k)
open(sys.argv[2], 'wb').write(c)
