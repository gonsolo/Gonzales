# pbrt-v4's AnimatedTransform for moving instances: matrices are 16 floats, column-major (flat[col*4 + row]).
from std.math import sqrt, sin, cos, asin, abs, min, max
from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from gonzales.core.geometry import _is_real_ptr
from gonzales.scene.shapes.primitives import Instance, affine_inverse

comptime M64 = SIMD[DType.float64, 16]
comptime M32 = SIMD[DType.float32, 16]
comptime Q32 = SIMD[DType.float32, 4]


@always_inline
def _mul3[dt: DType](a: SIMD[dt, 16], b: SIMD[dt, 16]) -> SIMD[dt, 16]:
    """a * b of the upper-left 3x3 blocks; the rest is identity."""
    var r = SIMD[dt, 16](0)
    r[15] = 1
    comptime for c in range(3):
        comptime for row in range(3):
            r[c * 4 + row] = a[row] * b[c * 4] + a[4 + row] * b[c * 4 + 1] + a[8 + row] * b[c * 4 + 2]
    return r


def _inv3(m: M64) -> M64:
    var a = m[0]; var b = m[4]; var c = m[8]
    var d = m[1]; var e = m[5]; var f = m[9]
    var g = m[2]; var h = m[6]; var i = m[10]
    var co00 = e * i - f * h; var co01 = f * g - d * i; var co02 = d * h - e * g
    var inv_det = 1.0 / (a * co00 + b * co01 + c * co02)
    var r = M64(0)
    r[0] = co00 * inv_det; r[4] = (c * h - b * i) * inv_det; r[8] = (b * f - c * e) * inv_det
    r[1] = co01 * inv_det; r[5] = (a * i - c * g) * inv_det; r[9] = (c * d - a * f) * inv_det
    r[2] = co02 * inv_det; r[6] = (b * g - a * h) * inv_det; r[10] = (a * e - b * d) * inv_det
    r[15] = 1
    return r


def _transpose3(m: M64) -> M64:
    var r = m
    comptime for c in range(3):
        comptime for row in range(3):
            r[c * 4 + row] = m[row * 4 + c]
    return r


def _quat_of(m: M64) -> SIMD[DType.float64, 4]:
    """(x, y, z, w) of a rotation matrix, as pbrt's Transform::operator Quaternion."""
    var trace = m[0] + m[5] + m[10]
    var q = SIMD[DType.float64, 4](0)
    if trace > 0:
        var s = sqrt(trace + 1.0)
        q[3] = s / 2
        s = 0.5 / s
        q[0] = (m[6] - m[9]) * s
        q[1] = (m[8] - m[2]) * s
        q[2] = (m[1] - m[4]) * s
        return q
    var i = 0
    if m[5] > m[0]: i = 1
    if m[10] > m[i * 5]: i = 2
    var j = (i + 1) % 3
    var k = (j + 1) % 3
    var s = sqrt(max(m[i * 5] - (m[j * 5] + m[k * 5]) + 1.0, 0.0))
    q[i] = s * 0.5
    if s != 0: s = 0.5 / s
    q[3] = (m[j * 4 + k] - m[k * 4 + j]) * s
    q[j] = (m[i * 4 + j] + m[j * 4 + i]) * s
    q[k] = (m[i * 4 + k] + m[k * 4 + i]) * s
    return q


def _decompose(m: M64) -> Tuple[SIMD[DType.float64, 4], SIMD[DType.float64, 4], M64]:
    """pbrt's Transform::Decompose: translation, rotation (as a quaternion) and the scale matrix."""
    var t = SIMD[DType.float64, 4](m[12], m[13], m[14], 0)
    var mm = m
    mm[3] = 0; mm[7] = 0; mm[11] = 0; mm[12] = 0; mm[13] = 0; mm[14] = 0; mm[15] = 1
    var r = mm
    for _ in range(100):
        var rnext = (r + _inv3(_transpose3(r))) * 0.5
        rnext[15] = 1
        var norm = 0.0
        for row in range(3):
            norm = max(norm, abs(r[row] - rnext[row]) + abs(r[4 + row] - rnext[4 + row]) + abs(r[8 + row] - rnext[8 + row]))
        r = rnext
        if norm <= 1e-4:
            break
    return (t, _quat_of(r), _mul3(_inv3(r), mm))


@always_inline
def _sin_x_over_x(x: Float32) -> Float32:
    if Float32(1) - x * x == Float32(1):
        return Float32(1)
    return sin(x) / x


@always_inline
def _angle_between(a: Q32, b: Q32) -> Float32:
    if (a * b).reduce_add() < Float32(0):
        var l = sqrt(((a + b) * (a + b)).reduce_add()) / Float32(2)
        return Float32(3.14159265358979) - Float32(2) * asin(min(l, Float32(1)))
    var l = sqrt(((b - a) * (b - a)).reduce_add()) / Float32(2)
    return Float32(2) * asin(min(l, Float32(1)))


@always_inline
def _rotation(q: Q32) -> M32:
    var x = q[0]; var y = q[1]; var z = q[2]; var w = q[3]
    var r = M32(0)
    r[0] = 1 - 2 * (y * y + z * z); r[4] = 2 * (x * y - w * z); r[8] = 2 * (x * z + w * y)
    r[1] = 2 * (x * y + w * z); r[5] = 1 - 2 * (x * x + z * z); r[9] = 2 * (y * z - w * x)
    r[2] = 2 * (x * z - w * y); r[6] = 2 * (y * z + w * x); r[10] = 1 - 2 * (x * x + y * y)
    r[15] = 1
    return r


@always_inline
def _mul4(a: M32, b: M32) -> M32:
    var r = M32(0)
    comptime for c in range(4):
        comptime for row in range(4):
            r[c * 4 + row] = a[row] * b[c * 4] + a[4 + row] * b[c * 4 + 1] + a[8 + row] * b[c * 4 + 2] + a[12 + row] * b[c * 4 + 3]
    return r


struct InstanceMotion(TrivialRegisterPassable):
    """One moving instance: the decomposed start and end placements and the map from template space into the space
    they act on (inverse of the CTM the template was baked with)."""
    var t0: Q32
    var t1: Q32
    var q0: Q32
    var q1: Q32
    var s0: M32
    var s1: M32
    var base: M32

    def __init__(out self, start: M64, end: M64, base: M64):
        var (t0, q0, s0) = _decompose(start)
        var (t1, q1, s1) = _decompose(end)
        if (q0 * q1).reduce_add() < 0:
            q1 = -q1
        self.t0 = t0.cast[DType.float32](); self.t1 = t1.cast[DType.float32]()
        self.q0 = q0.cast[DType.float32](); self.q1 = q1.cast[DType.float32]()
        self.s0 = s0.cast[DType.float32](); self.s1 = s1.cast[DType.float32]()
        self.base = base.cast[DType.float32]()

    @always_inline
    def obj_to_world(self, dt: Float32) -> M32:
        """pbrt's AnimatedTransform::Interpolate at dt in [0, 1], applied after `base`."""
        var theta = _angle_between(self.q0, self.q1)
        var sto = _sin_x_over_x(theta)
        var q = (self.q0 * (1 - dt) * _sin_x_over_x((1 - dt) * theta) + self.q1 * dt * _sin_x_over_x(dt * theta)) / sto
        var m = _mul4(_rotation(q), self.s0 * (1 - dt) + self.s1 * dt)
        var t = self.t0 * (1 - dt) + self.t1 * dt
        m[12] = t[0]; m[13] = t[1]; m[14] = t[2]
        return _mul4(m, self.base)

    def rotation_angle(self) -> Float32:
        return Float32(2) * _angle_between(self.q0, self.q1)


@fieldwise_init
struct MotionTable(TrivialRegisterPassable, DevicePassable):
    """The scene's moving instances (`index`: instance -> record, -1 static), pbrt's TransformTimes [t0, t1] and the
    time a path places them at. count == 0: nothing moves."""
    comptime device_type: AnyType = Self
    var index: Pointer[Int32, MutUntrackedOrigin]
    var records: Pointer[InstanceMotion, MutUntrackedOrigin]
    var count: Int32
    var t0: Float32
    var t1: Float32
    var time: Float32

    def _to_device_type(self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        return "MotionTable"

    @staticmethod
    def none() -> MotionTable:
        return MotionTable(Pointer[Int32, MutUntrackedOrigin].unsafe_dangling(),
                           Pointer[InstanceMotion, MutUntrackedOrigin].unsafe_dangling(),
                           Int32(0), Float32(0), Float32(1), Float32(0))

    @always_inline
    def at(self, time: Float32) -> MotionTable:
        var m = self
        m.time = time
        return m

    @always_inline
    def dt(self) -> Float32:
        if self.t1 > self.t0:
            return min(max((self.time - self.t0) / (self.t1 - self.t0), Float32(0)), Float32(1))
        return Float32(0)

    @always_inline
    def instance(self, instances: Pointer[Instance, MutUntrackedOrigin], k: Int) -> Instance:
        """Instance k placed at this table's time."""
        var inst = instances[unsafe_offset=k]
        if self.count == Int32(0):
            return inst
        var mi = self.index[unsafe_offset=k]
        if mi < Int32(0):
            return inst
        return Instance(affine_inverse(self.records[unsafe_offset=Int(mi)].obj_to_world(self.dt())), inst.blasIdx)
