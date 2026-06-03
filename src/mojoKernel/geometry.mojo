from std.ffi import external_call
from std.memory import alloc

@fieldwise_init
struct RGB(TrivialRegisterPassable):
    var r: Float32
    var g: Float32
    var b: Float32

    @always_inline
    fn __mul__(self, o: RGB) -> RGB:
        return RGB(self.r * o.r, self.g * o.g, self.b * o.b)

    @always_inline
    fn __mul__(self, s: Float32) -> RGB:
        return RGB(self.r * s, self.g * s, self.b * s)

    @always_inline
    fn __imul__(mut self, o: RGB):
        self.r *= o.r; self.g *= o.g; self.b *= o.b

    @always_inline
    fn __imul__(mut self, s: Float32):
        self.r *= s; self.g *= s; self.b *= s

    @always_inline
    fn __iadd__(mut self, o: RGB):
        self.r += o.r; self.g += o.g; self.b += o.b

    @always_inline
    fn luma(self) -> Float32:
        return Float32(0.2126)*self.r + Float32(0.7152)*self.g + Float32(0.0722)*self.b

@fieldwise_init
struct PrimId_C(TrivialRegisterPassable):
    var id1: Int64
    var id2: Int64
    var materialIndex: Int64
    var type: Int8
    var _pad0: Int8
    var _pad1: Int8
    var _pad2: Int8
    var _pad3: Int8
    var _pad4: Int8
    var _pad5: Int8
    var _pad6: Int8

@fieldwise_init
struct Material_C(TrivialRegisterPassable):
    var type: Int8
    var _pad0: Int8
    var _pad1: Int8
    var _pad2: Int8
    var albedo: RGB
    var emission: RGB
    var tex_idx: Int32   # -1 = no texture; >= 0 = index into texture table

@fieldwise_init
struct TriangleMesh_C(TrivialRegisterPassable):
    var points: UnsafePointer[Float32, MutAnyOrigin]
    var faceIndices: UnsafePointer[Int64, MutAnyOrigin]
    var vertexIndices: UnsafePointer[Int64, MutAnyOrigin]
    var uvs: UnsafePointer[Float32, MutAnyOrigin]   # nullable; stride 2 floats per vertex

@fieldwise_init
struct Ray_C(TrivialRegisterPassable):
    var orgX: Float32
    var orgY: Float32
    var orgZ: Float32
    var dirX: Float32
    var dirY: Float32
    var dirZ: Float32

@fieldwise_init
struct Intersection_C(TrivialRegisterPassable):
    var primId: PrimId_C
    var tHit: Float32
    var u: Float32
    var v: Float32
    var hit: Int8
    var _pad0: Int8
    var _pad1: Int8
    var _pad2: Int8

@fieldwise_init
struct PathState_C(TrivialRegisterPassable):
    var ray: Ray_C
    var throughput: RGB
    var estimate: RGB
    var albedo: RGB
    var bounce: Int32
    var pcgState: UInt64
    var pcgInc: UInt64
    var active: Int8
    var _pad1: Int8
    var _pad2: Int8
    var _pad3: Int8
    var _pad4: Int8
    var _pad5: Int8
    var _pad6: Int8
    var _pad7: Int8

@fieldwise_init
struct AreaLight_C(TrivialRegisterPassable):
    var meshIdx: Int32
    var n_tris: Int32       # number of triangles in this light mesh
    var emission: RGB
    var total_area: Float32 # total surface area of this light mesh

@fieldwise_init
struct PixelSample_C(TrivialRegisterPassable):
    var filmX: Float32
    var filmY: Float32
    var filterWeight: Float32
    var pixelX: Int32
    var pixelY: Int32
    var _pad: Int32
    var pcgState: UInt64
    var pcgInc: UInt64

@fieldwise_init
struct TileResult_C(TrivialRegisterPassable):
    var estimate: RGB
    var albedo: RGB
    var filterWeight: Float32
    var pixelX: Int32
    var pixelY: Int32

@always_inline
fn cross(a: SIMD[DType.float32, 3], b: SIMD[DType.float32, 3]) -> SIMD[DType.float32, 3]:
    var a_yzx = SIMD[DType.float32, 3](a[1], a[2], a[0])
    var b_zxy = SIMD[DType.float32, 3](b[2], b[0], b[1])
    var a_zxy = SIMD[DType.float32, 3](a[2], a[0], a[1])
    var b_yzx = SIMD[DType.float32, 3](b[1], b[2], b[0])
    return a_yzx * b_zxy - a_zxy * b_yzx

@always_inline
fn dot(a: SIMD[DType.float32, 3], b: SIMD[DType.float32, 3]) -> Float32:
    var prod = a * b
    return prod[0] + prod[1] + prod[2]

@always_inline
fn intersect_triangle(
    ray_org: SIMD[DType.float32, 3],
    ray_dir: SIMD[DType.float32, 3],
    p0: SIMD[DType.float32, 3],
    p1: SIMD[DType.float32, 3],
    p2: SIMD[DType.float32, 3],
    tMax: Float32
) -> Tuple[Bool, Float32, Float32, Float32]:
    var e1 = p1 - p0
    var e2 = p2 - p0
    var pvec = cross(ray_dir, e2)
    var det = dot(e1, pvec)

    if det > -0.0000001 and det < 0.0000001:
        return (False, tMax, 0.0, 0.0)

    var invDet = 1.0 / det
    var tvec = ray_org - p0
    var u = dot(tvec, pvec) * invDet

    if u < 0.0 or u > 1.0:
        return (False, tMax, 0.0, 0.0)

    var qvec = cross(tvec, e1)
    var v = dot(ray_dir, qvec) * invDet

    if v < 0.0 or u + v > 1.0:
        return (False, tMax, 0.0, 0.0)

    var t = dot(e2, qvec) * invDet
    if t <= 0.0 or t > tMax:
        return (False, tMax, 0.0, 0.0)

    return (True, t, u, v)
