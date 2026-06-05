# Geometry and Transformations

Every renderer needs a way to describe points in space, directions of travel,
and the volumes that enclose objects. Gonzales builds these from a small set
of plain structs in `geometry.mojo`.

## Math Constants

All formulas use named aliases rather than inline literals — every expression
reads like the paper it came from:

<!-- <<listing: Math Constants>> -->

## Points and Vectors

Gonzales distinguishes affine *points* (positions in space) from free *vectors*
(directions and displacements), following the same convention as pbrt-v4.
Both are 12-byte packed structs — identical in memory, distinct in meaning.

### Point3f

<!-- <<listing: Point3f>> -->

Point minus point yields a displacement vector (`__sub__`).
Point plus vector yields a new point (`__add__`).

### Vec3f

<!-- <<listing: Vec3f>> -->

`Vec3f` carries the full arithmetic set needed by shaders: addition,
subtraction, scalar multiply/divide, `length()`, `normalize()`, and a
member `dot()`. Using named structs instead of `SIMD[f32, 3]` avoids
the hidden power-of-2 lane padding that would bloat BVH nodes from 32
to 40 bytes, breaking the one-node-per-cache-line layout.

## Color and Spectra

<!-- <<listing: RGB>> -->

All arithmetic is in *scene-linear* space. The `luma()` method returns the
CIE Y luminance (Rec. 709 primaries), used for MIS PDF heuristics.

`SampledSpectrum` is currently a type alias for `RGB`:

<!-- <<listing: SampledSpectrum>> -->

This one-line alias makes every shader forward-compatible with a future
hero-wavelength spectral representation — swapping the alias is the only
change needed.

## Rays

<!-- <<listing: Ray_C>> -->

The `_C` suffix marks C-ABI-compatible structs that cross the Mojo↔C FFI
boundary and can live in GPU-visible flat buffers.

## Path State

Each in-flight path carries its full state in a flat `PathState_C` struct.
This layout allows arrays of paths to be stored in contiguous GPU buffers,
enabling wavefront processing where a single GPU kernel handles all active
paths at the same bounce depth in parallel.

<!-- <<listing: PathState_C>> -->

## Local Shading Frame

Many BSDF operations are most natural in a *local coordinate system* where
the shading normal points along +z. The `Frame` struct wraps an orthonormal
basis and provides `to_local` / `to_world` conversions.

Construction uses the **Duff et al. 2017** method
("Building an Orthonormal Basis, Revisited", JCGT 6(1)):

<!-- <<listing: Frame>> -->

The sign of `n.z` makes the formula branchless and numerically stable even
at the south pole (`n ≈ (0, 0, -1)`), avoiding the warp divergence that a
branch would cause on GPU.

## Geometry Helper Functions

### safe_sqrt

<!-- <<listing: safe_sqrt>> -->

Prevents NaN from small negative values caused by floating-point rounding
in expressions like `1 - cos²θ`.

### reflect

<!-- <<listing: reflect>> -->

### refract

<!-- <<listing: refract>> -->

Returns `(False, _)` on total internal reflection (when `sin²θₜ ≥ 1`).

### schlick_fresnel

<!-- <<listing: schlick_fresnel>> -->

`f0 = ((η−1)/(η+1))²` is the reflectance at normal incidence.

## Bounding Boxes and BVH

`BVH2Node` stores its axis-aligned bounding box as two `Point3f` corners.
At 32 bytes per node (12 + 12 + 4 + 4), two nodes fit in one 64-byte cache
line — the traversal hot-loop fetches both children in a single cache miss.
See [03_shapes_and_acceleration.md](03_shapes_and_acceleration.md) for details.

## Transforms

`transform.mojo` provides 4×4 column-major matrix math for camera and object
transforms. Matrices are stored as 16 `Float32` values following the
convention `flat[col*4 + row]`, matching the column-major layout used by
PBRT and OpenGL.
