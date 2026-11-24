"""The U-Net port: a small random network written as a .tza, run on the GPU, against a direct CPU evaluation of
OIDN's unet_large graph (training/model.py) with the same input and output transforms."""
from std.math import abs, max
from std.memory import bitcast
from std.testing import assert_true, TestSuite
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from gonzales.integrators.unet import UNetWeights, unet_denoise, autoexposure, pu_forward, pu_inverse

comptime C = 8
comptime W = 40
comptime H = 24
comptime PW = 48
comptime PH = 32
comptime PATH = "/tmp/gonzales_test_unet.tza"


def _layers() -> List[Tuple[String, Int, Int]]:
    return [("enc_conv1a", 9, C), ("enc_conv1b", C, C), ("enc_conv2a", C, C), ("enc_conv2b", C, C),
            ("enc_conv3a", C, C), ("enc_conv3b", C, C), ("enc_conv4a", C, C), ("enc_conv4b", C, C),
            ("enc_conv5a", C, C), ("enc_conv5b", C, C), ("dec_conv4a", 2 * C, C), ("dec_conv4b", C, C),
            ("dec_conv3a", 2 * C, C), ("dec_conv3b", C, C), ("dec_conv2a", 2 * C, C), ("dec_conv2b", C, C),
            ("dec_conv1a", C + 9, C), ("dec_conv1b", C, C), ("dec_conv1c", C, 3)]


def _val(i: Int) -> Float32:
    """Pseudo-random, exact in float16."""
    var x = (UInt64(i) * 2654435761 + 12345) % 4294967296
    return Float32(Int((x >> 13) % 33) - 16) / 64


def _put(mut f: List[UInt8], v: Int, n: Int):
    for k in range(n):
        f.append(UInt8((v >> (8 * k)) & 255))


def _write_tza() raises -> List[List[Float32]]:
    """Weights (oihw) and biases of every layer, alternating; also written to PATH."""
    var f = List[UInt8](); var t = List[List[Float32]](); var table = List[Tuple[String, List[Int], Int]]()
    _put(f, 0x41D7, 2); _put(f, 2, 1); _put(f, 0, 1); _put(f, 0, 8)
    var seed = 0
    for l in _layers():
        for is_bias in range(2):
            var shape: List[Int] = [l[2]] if is_bias else [l[2], l[1], 3, 3]
            var n = l[2] if is_bias else l[2] * l[1] * 9
            while len(f) % 64 != 0:
                f.append(0)
            table.append((l[0] + (".bias" if is_bias else ".weight"), shape^, len(f)))
            var v = List[Float32]()
            for i in range(n):
                v.append(_val(seed + i) * (Float32(0.5) if is_bias else Float32(1)))
                _put(f, Int(bitcast[DType.uint16, 1](v[i].cast[DType.float16]())), 2)
            seed += n
            t.append(v^)
    while len(f) % 64 != 0:
        f.append(0)
    var table_off = len(f)
    _put(f, len(table), 4)
    for e in table:
        _put(f, e[0].byte_length(), 2)
        for b in e[0].as_bytes():
            f.append(b)
        _put(f, len(e[1]), 1)
        for d in e[1]:
            _put(f, d, 4)
        for b in String("x" if len(e[1]) == 1 else "oihw").as_bytes():
            f.append(b)
        f.append(UInt8(ord("h")))
        _put(f, e[2], 8)
    for k in range(8):
        f[4 + k] = UInt8((table_off >> (8 * k)) & 255)
    var fh = open(PATH, "w")
    fh.write_bytes(Span(f))
    fh.close()
    return t^


def _conv(x: List[Float32], ci: Int, w: Int, h: Int, wt: List[Float32], b: List[Float32], co: Int) -> List[Float32]:
    var y = List[Float32](length=co * w * h, fill=0)
    for o in range(co):
        for py in range(h):
            for px in range(w):
                var acc = b[o]
                for i in range(ci):
                    for ky in range(3):
                        for kx in range(3):
                            var sx = px + kx - 1; var sy = py + ky - 1
                            if sx >= 0 and sx < w and sy >= 0 and sy < h:
                                acc += x[(i * h + sy) * w + sx] * wt[((o * ci + i) * 3 + ky) * 3 + kx]
                y[(o * h + py) * w + px] = max(acc, Float32(0))
    return y^


def _pool(x: List[Float32], c: Int, w: Int, h: Int) -> List[Float32]:
    var y = List[Float32](length=c * w * h // 4, fill=0)
    for k in range(c):
        for py in range(h // 2):
            for px in range(w // 2):
                var m = x[(k * h + 2 * py) * w + 2 * px]
                m = max(m, x[(k * h + 2 * py) * w + 2 * px + 1])
                m = max(m, x[(k * h + 2 * py + 1) * w + 2 * px])
                m = max(m, x[(k * h + 2 * py + 1) * w + 2 * px + 1])
                y[(k * (h // 2) + py) * (w // 2) + px] = m
    return y^


def _up_cat(a: List[Float32], ca: Int, b: List[Float32], cb: Int, w: Int, h: Int) -> List[Float32]:
    """concat(upsample(a), b) at b's size w x h."""
    var y = List[Float32](length=(ca + cb) * w * h, fill=0)
    for k in range(ca):
        for py in range(h):
            for px in range(w):
                y[(k * h + py) * w + px] = a[(k * (h // 2) + py // 2) * (w // 2) + px // 2]
    for i in range(cb * w * h):
        y[ca * w * h + i] = b[i]
    return y^


def _reference(t: List[List[Float32]], col: List[Float32], alb: List[Float32], nrm: List[Float32]) -> List[Float32]:
    var e = autoexposure(col.unsafe_ptr(), W, H)
    var norm = Float32(1) / pu_forward(Float32(65504))
    var x = List[Float32](length=9 * PW * PH, fill=0)
    for py in range(H):
        for px in range(W):
            for c in range(3):
                var s = (py * W + px) * 3 + c; var d = py * PW + px
                x[c * PW * PH + d] = pu_forward(col[s] * e) * norm
                x[(3 + c) * PW * PH + d] = alb[s]
                x[(6 + c) * PW * PH + d] = nrm[s] * Float32(0.5) + Float32(0.5)
    var w = PW; var h = PH
    var a = _conv(x, 9, w, h, t[0], t[1], C)
    var p1 = _pool(_conv(a, C, w, h, t[2], t[3], C), C, w, h)
    w //= 2; h //= 2
    var p2 = _pool(_conv(_conv(p1, C, w, h, t[4], t[5], C), C, w, h, t[6], t[7], C), C, w, h)
    w //= 2; h //= 2
    var p3 = _pool(_conv(_conv(p2, C, w, h, t[8], t[9], C), C, w, h, t[10], t[11], C), C, w, h)
    w //= 2; h //= 2
    var p4 = _pool(_conv(_conv(p3, C, w, h, t[12], t[13], C), C, w, h, t[14], t[15], C), C, w, h)
    w //= 2; h //= 2
    var y = _conv(_conv(p4, C, w, h, t[16], t[17], C), C, w, h, t[18], t[19], C)
    w *= 2; h *= 2
    y = _conv(_conv(_up_cat(y, C, p3, C, w, h), 2 * C, w, h, t[20], t[21], C), C, w, h, t[22], t[23], C)
    w *= 2; h *= 2
    y = _conv(_conv(_up_cat(y, C, p2, C, w, h), 2 * C, w, h, t[24], t[25], C), C, w, h, t[26], t[27], C)
    w *= 2; h *= 2
    y = _conv(_conv(_up_cat(y, C, p1, C, w, h), 2 * C, w, h, t[28], t[29], C), C, w, h, t[30], t[31], C)
    w *= 2; h *= 2
    y = _conv(_conv(_up_cat(y, C, x, 9, w, h), C + 9, w, h, t[32], t[33], C), C, w, h, t[34], t[35], C)
    y = _conv(y, C, w, h, t[36], t[37], 3)
    var res = List[Float32](length=W * H * 3, fill=0)
    for py in range(H):
        for px in range(W):
            for c in range(3):
                res[(py * W + px) * 3 + c] = pu_inverse(y[c * PW * PH + py * PW + px] / norm) / e
    return res^


def test_pu_round_trip() raises:
    var y = Float32(1e-7)
    while y < Float32(6e4):
        assert_true(abs(pu_inverse(pu_forward(y)) - y) <= Float32(2e-4) * y, String("PU round trip at ") + String(y))
        y *= Float32(1.7)


def test_unet_matches_reference() raises:
    comptime if not has_accelerator():
        return
    var t = _write_tza()
    var col = List[Float32](); var alb = List[Float32](); var nrm = List[Float32]()
    for i in range(W * H * 3):
        col.append((_val(7 * i + 1) + Float32(0.3)) * Float32(4) if i % 97 else Float32(40))
        col[i] = max(col[i], Float32(0))
        alb.append(_val(5 * i + 2) * Float32(2) + Float32(0.5))
        nrm.append(_val(3 * i + 3) * Float32(4))
    var want = _reference(t, col, alb, nrm)
    var got = List[Float32](length=W * H * 3, fill=0)
    unet_denoise(DeviceContext(), UNetWeights(PATH), col.unsafe_ptr(), alb.unsafe_ptr(), nrm.unsafe_ptr(),
                 got.unsafe_ptr(), W, H)
    var worst = Float32(0); var lit = 0
    for i in range(W * H * 3):
        worst = max(worst, abs(got[i] - want[i]) / (abs(want[i]) + Float32(1e-3)))
        if want[i] > Float32(0):
            lit += 1
    assert_true(lit > W * H, String("the reference network is mostly dark: ") + String(lit))
    assert_true(worst < Float32(1e-3), String("GPU U-Net off the reference by ") + String(worst))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
