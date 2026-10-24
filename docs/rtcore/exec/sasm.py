"""Minimal sm_86 assembler: driver-code templates with fields replaced; every word is checked with nvdisasm."""
import subprocess, tempfile, re
import os
_c = open(os.environ.get('SASM_CODE', 'code.bin'), 'rb').read()      # the driver's intersect_batch code (pipeline_dump)
def _t(src, o): return int.from_bytes(src[o:o + 16], 'little')
T = dict(movi=_t(_c, 0x2a0), movr=_t(_c, 0x4d0), iaddi=_t(_c, 0x70), shl=_t(_c, 0x50),
         isetp=_t(_c, 0x3b0), bra=_t(_c, 0x240), exit=_t(_c, 0x40), lop3pi=_t(_c, 0x370))
CTRL = 8 | 1 << 4 | 7 << 5 | 7 << 8 | 0x3f << 11            # stall 8 (12 faults), yield, no barriers, wait on all six
def _set(w, lo, n, v): return (w & ~(((1 << n) - 1) << lo)) | ((v & ((1 << n) - 1)) << lo)
def _fin(w, pred=7, neg=0, ctrl=CTRL):
    w = _set(w, 12, 3, pred); w = _set(w, 15, 1, neg); return _set(w, 105, 23, ctrl)
def R(s): return 255 if s == 'RZ' else int(s[1:])
def movi(d, imm, **k): return _fin(_set(_set(T['movi'], 16, 8, d), 32, 32, imm), **k)
def movr(d, s, **k): return _fin(_set(_set(T['movr'], 16, 8, d), 64, 8, s), **k)
def iaddi(d, a, imm, **k): return _fin(_set(_set(_set(T['iaddi'], 16, 8, d), 24, 8, a), 32, 32, imm), **k)
def iaddr(d, a, b, **k): return _fin(_set(_set(_set(_set(_set(T['iaddi'], 9, 3, 1), 16, 8, d), 24, 8, a), 32, 32, 0), 32, 8, b), **k)
def shl(d, a, imm, **k): return _fin(_set(_set(_set(T['shl'], 16, 8, d), 24, 8, a), 32, 32, imm), **k)
def isetp_ne(p, a, b, **k): return _fin(_set(_set(_set(T['isetp'], 81, 3, p), 24, 8, a), 32, 8, b), **k)
def bra(at, target, **k): return _fin(_set(T['bra'], 32, 50, target - at - 16), **k)
def exit_(**k): return _fin(T['exit'], **k)
def raw(w): return w
def check(words, base):
    f = tempfile.NamedTemporaryFile(suffix='.bin', delete=False)
    f.write(b''.join(w.to_bytes(16, 'little') for w in words)); f.close()
    txt = subprocess.run(['nvdisasm', '--binary', 'SM86', '--base-address', hex(base), f.name], capture_output=True, text=True).stdout
    return [re.sub(r'\s+', ' ', l.split('*/', 1)[1]).strip() for l in txt.splitlines() if re.match(r'\s+/\*[0-9a-f]{4}\*/', l)]
T['stg'] = _t(_c, 0x12e0)
STG_CTRL = 8 | 1 << 4 | 7 << 5 | 0 << 8 | 0x3f << 11       # sets read barrier 0
def stg(a, b, **k): return _fin(_set(_set(T['stg'], 24, 8, a), 32, 8, b), ctrl=STG_CTRL, **k)
def patch(code, at, words):
    for i, w in enumerate(words): code[at + 16 * i: at + 16 * i + 16] = w.to_bytes(16, 'little')
    return at + 16 * len(words)
FLAGS = [(20, 0xff030207), (21, 0), (22, 0xff0000), (23, 0), (24, 0xff030407)]
if __name__ == '__main__':
    ws = [movi(20, 0xff030207), movr(30, 8), iaddi(31, 26, 0x14), iaddr(30, 30, 47), shl(30, 26, 0x10),
          isetp_ne(1, 36, 255), bra(0x840, 0x6d0, pred=1), bra(0x850, 0x900), exit_()]
    for l in check(ws, 0x7f0): print(l)
