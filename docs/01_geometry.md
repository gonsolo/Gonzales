# Geometry and Transformations

Every renderer needs a way to describe points in space, directions of travel,
and the volumes that enclose objects. Gonzales builds these from a small set
of plain structs in `geometry.mojo`.

## Vectors and Color

The primary color/radiance type is `RGB`, storing three `Float32` components:

```mojo
struct RGB(TrivialRegisterPassable):
    var r: Float32
    var g: Float32
    var b: Float32
```

Arithmetic operators (`*`, `+=`, `*=`) are defined directly on the struct.
`TrivialRegisterPassable` tells Mojo the value can be passed in registers
and stored in GPU-friendly flat buffers without boxing.

## Rays

A ray is an origin point plus a normalized direction, stored as six floats:

```mojo
struct Ray_C(TrivialRegisterPassable):
    var orgX: Float32; var orgY: Float32; var orgZ: Float32
    var dirX: Float32; var dirY: Float32; var dirZ: Float32
```

The `_C` suffix marks C-ABI-compatible structs that cross the Mojo↔C FFI
boundary.

## Path State

Each in-flight path carries its full state in a flat `PathState_C` struct.
This layout allows arrays of paths to be stored in contiguous GPU buffers:

```mojo
struct PathState_C(TrivialRegisterPassable):
    var ray: Ray_C
    var throughput: RGB    # weight accumulated along the path
    var estimate: RGB      # radiance accumulated so far
    var albedo: RGB        # first-bounce surface color (for denoiser)
    var bounce: Int32
    var pcgState: UInt64   # PCG32 random state
    var pcgInc: UInt64
    var active: Int8       # 0 = terminated
```

Keeping path state flat and fixed-size is the key that allows wavefront
GPU processing: a single GPU kernel can process all active paths at the
same bounce depth in parallel.

## Bounding Boxes

Axis-aligned bounding boxes (`Bounds_C`) enclose every object. The BVH
traversal tests rays against these boxes using the slab method: compute
entry and exit distances for each axis, then check whether there is an
overlap. The precomputed inverse direction avoids three divisions per test.

## Transforms

`transform.mojo` provides 4×4 column-major matrix math for camera and
object transforms. Matrices are stored as 16 `Float32` values following
the convention `flat[col*4 + row]`, matching the column-major layout used
by PBRT and OpenGL.
