"""The denoising U-Net: OIDN's unet_large (training/model.py), run with weights we trained on our own renders (.tza).
Inputs are the noisy HDR image, the albedo and the normal; the transfer function and autoexposure match OIDN's."""
from std.math import ceildiv, log, exp, pow, log2, exp2
from std.memory import bitcast
from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceContext, DeviceBuffer

comptime _PU_A = Float32(1.41283765e+03)
comptime _PU_B = Float32(1.64593172e+00)
comptime _PU_C = Float32(4.31384981e-01)
comptime _PU_D = Float32(-2.94139609e-03)
comptime _PU_E = Float32(1.92653254e-01)
comptime _PU_F = Float32(6.26026094e-03)
comptime _PU_G = Float32(9.98620152e-01)
comptime _PU_Y0 = Float32(1.57945760e-06)
comptime _PU_Y1 = Float32(3.22087631e-02)
comptime _PU_X0 = Float32(2.23151711e-03)
comptime _PU_X1 = Float32(3.70974749e-01)
comptime _HDR_MAX = Float32(65504.0)
comptime _FLT_MAX = Float32(3.4028234e38)
comptime _OB = 16      # output channels per thread
comptime _BLOCK = 256

def _layers() -> List[String]:
    return [
        "enc_conv1a", "enc_conv1b", "enc_conv2a", "enc_conv2b", "enc_conv3a", "enc_conv3b", "enc_conv4a", "enc_conv4b",
        "enc_conv5a", "enc_conv5b", "dec_conv4a", "dec_conv4b", "dec_conv3a", "dec_conv3b", "dec_conv2a", "dec_conv2b",
        "dec_conv1a", "dec_conv1b", "dec_conv1c",
    ]


@always_inline
def pu_forward(y: Float32) -> Float32:
    if y <= _PU_Y0:
        return _PU_A * y
    if y <= _PU_Y1:
        return _PU_B * pow(y, _PU_C) + _PU_D
    return _PU_E * log(y + _PU_F) + _PU_G


@always_inline
def pu_inverse(x: Float32) -> Float32:
    if x <= _PU_X0:
        return x / _PU_A
    if x <= _PU_X1:
        return pow((x - _PU_D) / _PU_B, Float32(1) / _PU_C)
    return exp((x - _PU_G) / _PU_E) - _PU_F


@always_inline
def _clean(v: Float32, lo: Float32, hi: Float32) -> Float32:
    return min(max(v, lo), hi) if v == v else Float32(0)


struct UNetWeights(Movable):
    """Per layer: weights as [ic][ky*3+kx][oc padded to _OB], then the padded bias."""
    var data: List[Float32]
    var w_off: List[Int]
    var b_off: List[Int]
    var ci: List[Int]
    var co: List[Int]

    def __init__(out self, path: String) raises:
        var f = open(path, "r")
        var raw = f.read_bytes()
        f.close()
        self.data = List[Float32]()
        self.w_off = List[Int](); self.b_off = List[Int](); self.ci = List[Int](); self.co = List[Int]()
        if _u(raw, 0, 2) != 0x41D7:
            raise Error("not a .tza file: " + path)
        var pos = _u(raw, 4, 8)
        var n = _u(raw, pos, 4); pos += 4
        var names = List[String](); var shapes = List[List[Int]](); var offs = List[Int]()
        for _ in range(n):
            var ln = _u(raw, pos, 2); pos += 2
            var name = String(StringSlice(unsafe_from_utf8=Span(raw)[pos:pos + ln])); pos += ln
            var nd = _u(raw, pos, 1); pos += 1
            var shape = List[Int]()
            for _ in range(nd):
                shape.append(_u(raw, pos, 4)); pos += 4
            pos += nd
            if raw[pos] != UInt8(ord("h")):
                raise Error("tza: only float16 tensors: " + name)
            pos += 1
            offs.append(_u(raw, pos, 8)); pos += 8
            names.append(name^); shapes.append(shape^)
        for layer in _layers():
            var wi = _find(names, layer + ".weight"); var bi = _find(names, layer + ".bias")
            var co = shapes[wi][0]; var ci = shapes[wi][1]
            var cop = ceildiv(co, _OB) * _OB
            self.ci.append(ci); self.co.append(co)
            self.w_off.append(len(self.data))
            self.data.resize(len(self.data) + ci * 9 * cop, Float32(0))
            for o in range(co):
                for i in range(ci):
                    for k in range(9):
                        self.data[self.w_off[len(self.w_off) - 1] + (i * 9 + k) * cop + o] = _half(raw, offs[wi] + ((o * ci + i) * 9 + k) * 2)
            self.b_off.append(len(self.data))
            self.data.resize(len(self.data) + cop, Float32(0))
            for o in range(co):
                self.data[self.b_off[len(self.b_off) - 1] + o] = _half(raw, offs[bi] + o * 2)


def _u(raw: List[UInt8], pos: Int, n: Int) -> Int:
    var v = 0
    for i in range(n):
        v |= Int(raw[pos + i]) << (8 * i)
    return v


def _half(raw: List[UInt8], pos: Int) -> Float32:
    return bitcast[DType.float16, 1](UInt16(_u(raw, pos, 2))).cast[DType.float32]()


def _find(names: List[String], name: String) raises -> Int:
    for i in range(len(names)):
        if names[i] == name:
            return i
    raise Error("tza: no tensor " + name)


def autoexposure(color: Pointer[Float32, _], w: Int, h: Int) -> Float32:
    """OIDN's: 0.18 over the geometric mean luminance of 16x16-pixel bin means."""
    var nbh = ceildiv(h, 16); var nbw = ceildiv(w, 16)
    var sum = Float64(0); var cnt = 0
    for i in range(nbh):
        for j in range(nbw):
            var y0 = i * h // nbh; var y1 = (i + 1) * h // nbh; var x0 = j * w // nbw; var x1 = (j + 1) * w // nbw
            var l = Float32(0)
            for y in range(y0, y1):
                for x in range(x0, x1):
                    var p = (y * w + x) * 3
                    l += Float32(0.212671) * color[unsafe_offset=p] + Float32(0.715160) * color[unsafe_offset=p + 1] + \
                         Float32(0.072169) * color[unsafe_offset=p + 2]
            l /= Float32((y1 - y0) * (x1 - x0))
            if l > Float32(1e-8):
                sum += Float64(log2(l)); cnt += 1
    return Float32(0.18) / exp2(Float32(sum / Float64(cnt))) if cnt > 0 else Float32(1)


def unet_input_gpu(
    color: Pointer[Float32, MutUntrackedOrigin], albedo: Pointer[Float32, MutUntrackedOrigin],
    normal: Pointer[Float32, MutUntrackedOrigin], res: Pointer[Float32, MutUntrackedOrigin],
    w: Int32, h: Int32, pw: Int32, ph: Int32, exposure: Float32, norm: Float32,
):
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    var n = Int(pw) * Int(ph)
    if tid >= n:
        return
    var x = tid % Int(pw); var y = tid // Int(pw)
    var inside = x < Int(w) and y < Int(h)
    var s = (y * Int(w) + x) * 3
    for c in range(3):
        var v = Float32(0); var a = Float32(0); var nm = Float32(0)
        if inside:
            v = pu_forward(_clean(color[unsafe_offset=s + c] * exposure, 0, _FLT_MAX)) * norm
            a = _clean(albedo[unsafe_offset=s + c], 0, 1)
            nm = _clean(normal[unsafe_offset=s + c], -1, 1) * Float32(0.5) + Float32(0.5)
        res[unsafe_offset=c * n + tid] = v
        res[unsafe_offset=(3 + c) * n + tid] = a
        res[unsafe_offset=(6 + c) * n + tid] = nm


@always_inline
def _accum[UP: Bool](
    mut acc: SIMD[DType.float32, _OB], src: Pointer[Float32, MutUntrackedOrigin], c0: Int, c1: Int, wt_c0: Int,
    wt: Pointer[Float32, MutUntrackedOrigin], cop: Int, ob: Int, x: Int, y: Int, cw: Int, chh: Int,
):
    """Channels [c0, c1) of a planar tensor at the conv's size, or at half of it (nearest upsampling)."""
    var sw = cw // 2 if UP else cw
    var plane = sw * (chh // 2 if UP else chh)
    for c in range(c0, c1):
        var base = src.unsafe_offset((c - c0) * plane)
        for ky in range(3):
            var yy = y + ky - 1
            if yy < 0 or yy >= chh:
                continue
            var row = (yy // 2 if UP else yy) * sw
            for kx in range(3):
                var xx = x + kx - 1
                if xx < 0 or xx >= cw:
                    continue
                var v = base[unsafe_offset=row + (xx // 2 if UP else xx)]
                var wv = wt.unsafe_offset(((wt_c0 + c - c0) * 9 + ky * 3 + kx) * cop + ob * _OB).unsafe_load[width=_OB, alignment=64]()
                acc += v * wv


def unet_conv_gpu[POOL: Bool, UP: Bool](
    a: Pointer[Float32, MutUntrackedOrigin], ca: Int32, b: Pointer[Float32, MutUntrackedOrigin], cb: Int32,
    wt: Pointer[Float32, MutUntrackedOrigin], bias: Pointer[Float32, MutUntrackedOrigin],
    res: Pointer[Float32, MutUntrackedOrigin], co: Int32, co_pad: Int32, cw_i: Int32, ch_i: Int32,
):
    """3x3 conv + bias + ReLU over concat(up(a) if UP else a, b), max-pooled 2x2 if POOL; planar tensors."""
    var cw = Int(cw_i); var chh = Int(ch_i); var cop = Int(co_pad)
    var ow = cw // 2 if POOL else cw; var oh = chh // 2 if POOL else chh
    var npix = ow * oh
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= npix * (cop // _OB):
        return
    var ob = tid // npix; var p = tid % npix
    var ox = p % ow; var oy = p // ow
    var bv = bias.unsafe_offset(ob * _OB).unsafe_load[width=_OB, alignment=64]()
    var best = SIMD[DType.float32, _OB](0)
    comptime NS = 2 if POOL else 1
    for sy in range(NS):
        for sx in range(NS):
            var x = ox * NS + sx; var y = oy * NS + sy
            var acc = bv
            _accum[UP](acc, a, 0, Int(ca), 0, wt, cop, ob, x, y, cw, chh)
            _accum[False](acc, b, 0, Int(cb), Int(ca), wt, cop, ob, x, y, cw, chh)
            best = max(best, acc)    # ReLU: best starts at 0
    for o in range(min(_OB, Int(co) - ob * _OB)):
        res[unsafe_offset=(ob * _OB + o) * npix + p] = best[o]


def unet_output_gpu(
    src: Pointer[Float32, MutUntrackedOrigin], res: Pointer[Float32, MutUntrackedOrigin],
    w: Int32, h: Int32, pw: Int32, ph: Int32, inv_exposure: Float32, inv_norm: Float32,
):
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= Int(w) * Int(h):
        return
    var x = tid % Int(w); var y = tid // Int(w)
    var n = Int(pw) * Int(ph)
    for c in range(3):
        var v = _clean(src[unsafe_offset=c * n + y * Int(pw) + x], 0, _FLT_MAX)
        res[unsafe_offset=tid * 3 + c] = pu_inverse(v * inv_norm) * inv_exposure


def unet_denoise(
    ctx: DeviceContext, wts: UNetWeights, color: Pointer[Float32, _], albedo: Pointer[Float32, _],
    normal: Pointer[Float32, _], res: Pointer[mut=True, Float32, _], w: Int, h: Int,
) raises:
    """Denoises the w x h RGB `color` (HDR, linear) into `res`; albedo and normal are noise-free guides."""
    var pw = ceildiv(w, 16) * 16; var ph = ceildiv(h, 16) * 16
    var n = pw * ph
    var exposure = autoexposure(color, w, h)
    var norm = Float32(1) / pu_forward(_HDR_MAX)
    var d_col = ctx.enqueue_create_buffer[DType.float32](w * h * 3)
    var d_alb = ctx.enqueue_create_buffer[DType.float32](w * h * 3)
    var d_nrm = ctx.enqueue_create_buffer[DType.float32](w * h * 3)
    ctx.enqueue_copy(d_col, color.as_imm())
    ctx.enqueue_copy(d_alb, albedo.as_imm())
    ctx.enqueue_copy(d_nrm, normal.as_imm())
    var d_w = ctx.enqueue_create_buffer[DType.float32](len(wts.data))
    ctx.enqueue_copy(d_w, wts.data.unsafe_ptr().as_imm())
    var d_in = ctx.enqueue_create_buffer[DType.float32](9 * n)
    var p1 = ctx.enqueue_create_buffer[DType.float32](wts.co[1] * n // 4)
    var p2 = ctx.enqueue_create_buffer[DType.float32](wts.co[3] * n // 16)
    var p3 = ctx.enqueue_create_buffer[DType.float32](wts.co[5] * n // 64)
    var big = max(wts.co[0], wts.co[17]) * n
    var ta = ctx.enqueue_create_buffer[DType.float32](big)
    var tb = ctx.enqueue_create_buffer[DType.float32](big)
    ctx.enqueue_function[unet_input_gpu](
        _p(d_col), _p(d_alb), _p(d_nrm), _p(d_in), Int32(w), Int32(h), Int32(pw), Int32(ph), exposure, norm,
        grid_dim=ceildiv(n, _BLOCK), block_dim=_BLOCK)

    def conv[POOL: Bool, UP: Bool](l: Int, a: DeviceBuffer[DType.float32], b: DeviceBuffer[DType.float32],
                                   cb: Int, dst: DeviceBuffer[DType.float32], level: Int) raises {imm}:
        var cw = pw >> level; var chh = ph >> level
        var cop = ceildiv(wts.co[l], _OB) * _OB
        var npix = (cw // 2) * (chh // 2) if POOL else cw * chh
        var wp = _p(d_w).unsafe_offset(wts.w_off[l]); var bp = _p(d_w).unsafe_offset(wts.b_off[l])
        ctx.enqueue_function[unet_conv_gpu[POOL, UP]](
            _p(a), Int32(wts.ci[l] - cb), _p(b), Int32(cb), wp, bp, _p(dst), Int32(wts.co[l]), Int32(cop), Int32(cw),
            Int32(chh),
            grid_dim=ceildiv(npix * (cop // _OB), _BLOCK), block_dim=_BLOCK)

    conv[False, False](0, d_in, d_in, 0, ta, 0)          # enc_conv1a
    conv[True, False](1, ta, ta, 0, p1, 0)               # enc_conv1b, pool1
    conv[False, False](2, p1, p1, 0, ta, 1)
    conv[True, False](3, ta, ta, 0, p2, 1)               # pool2
    conv[False, False](4, p2, p2, 0, ta, 2)
    conv[True, False](5, ta, ta, 0, p3, 2)               # pool3
    conv[False, False](6, p3, p3, 0, ta, 3)
    conv[True, False](7, ta, ta, 0, tb, 3)               # pool4
    conv[False, False](8, tb, tb, 0, ta, 4)
    conv[False, False](9, ta, ta, 0, tb, 4)              # enc_conv5b
    conv[False, True](10, tb, p3, wts.co[5], ta, 3)      # upsample4 + concat pool3
    conv[False, False](11, ta, ta, 0, tb, 3)
    conv[False, True](12, tb, p2, wts.co[3], ta, 2)
    conv[False, False](13, ta, ta, 0, tb, 2)
    conv[False, True](14, tb, p1, wts.co[1], ta, 1)
    conv[False, False](15, ta, ta, 0, tb, 1)
    conv[False, True](16, tb, d_in, 9, ta, 0)            # upsample1 + concat input
    conv[False, False](17, ta, ta, 0, tb, 0)
    conv[False, False](18, tb, tb, 0, ta, 0)             # dec_conv1c
    ctx.enqueue_function[unet_output_gpu](
        _p(ta), _p(d_col), Int32(w), Int32(h), Int32(pw), Int32(ph), Float32(1) / exposure, Float32(1) / norm,
        grid_dim=ceildiv(w * h, _BLOCK), block_dim=_BLOCK)
    ctx.enqueue_copy(res.unsafe_origin_cast[MutUntrackedOrigin](), d_col)
    ctx.synchronize()


@always_inline
def _p(b: DeviceBuffer[DType.float32]) -> Pointer[Float32, MutUntrackedOrigin]:
    return b.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
