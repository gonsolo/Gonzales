# Geometry and Transformations

Every renderer needs a way to describe points in space, directions of travel,
and the volumes that enclose objects. Gonzales builds these from a small set
of value types, all backed by SIMD4 for vectorized arithmetic.

## Vectors

The fundamental geometric type is `Vector3`. It stores three `Real` components
in a `SIMD4<Real>`, which lets the compiler emit a single SIMD instruction
for operations like addition and scaling:

{{snippet:Sources/libgonzales/Geometry/Vector3.swift:vector3-struct}}

The fourth SIMD lane is unused but keeps the struct aligned to 32 bytes on
64-bit platforms, which is optimal for cache-line access patterns during
BVH traversal.

`Point3` and `Normal3` share the same SIMD4 backing but are distinct types —
a deliberate design choice that prevents accidentally adding a point to a
normal without an explicit conversion.

## Rays

A ray is an origin plus a direction. Gonzales also precomputes the
inverse direction, which avoids three divisions per bounding-box intersection
test — one of the hottest paths in the renderer:

{{snippet:Sources/libgonzales/Core/Ray.swift:ray-struct}}

The `getPointFor(parameter:)` method evaluates the parametric ray equation
**p(t) = o + t·d**, used everywhere from intersection tests to spawn points.

## Bounding Boxes

Axis-aligned bounding boxes (`Bounds3`) enclose every object in the scene.
The BVH acceleration structure tests rays against these boxes to skip
large groups of geometry. This intersection test is performance-critical —
it runs millions of times per frame:

{{snippet:Sources/libgonzales/Geometry/Bounds3.swift:bounds-intersect}}

The slab method computes entry and exit distances for each axis independently
using the precomputed inverse direction, then checks whether there is an
overlap. The `gamma(count:)` error correction accounts for floating-point
rounding to avoid missing intersections at grazing angles — the same
technique described in PBRT Section 3.9.
