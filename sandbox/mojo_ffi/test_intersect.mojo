@always_inline
fn cross(a: SIMD[DType.float32, 3], b: SIMD[DType.float32, 3]) -> SIMD[DType.float32, 3]:
    var a_yzx = SIMD[DType.float32, 3](a[1], a[2], a[0])
    var b_zxy = SIMD[DType.float32, 3](b[2], b[0], b[1])
    var a_zxy = SIMD[DType.float32, 3](a[2], a[0], a[1])
    var b_yzx = SIMD[DType.float32, 3](b[1], b[2], b[0])
    return a_yzx * b_zxy - a_zxy * b_yzx

@always_inline
fn dot(a: SIMD[DType.float32, 3], b: SIMD[DType.float32, 3]) -> Float32:
    return (a * b).reduce_add()

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
    if t < 0.0000001 or t > tMax:
        return (False, tMax, 0.0, 0.0)
        
    return (True, t, u, v)

fn main():
    var a = SIMD[DType.float32, 3](1, 0, 0)
    var b = SIMD[DType.float32, 3](0, 1, 0)
    _ = cross(a, b)
