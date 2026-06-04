# Shapes and Acceleration

A renderer spends most of its time answering one question: does this ray
hit anything? Gonzales supports triangle meshes, spheres, and PLY files,
but the key to performance is the BVH that avoids testing every shape.

## Shape Types

Triangle meshes are the dominant primitive. Each mesh stores its vertices,
face indices, vertex indices, and UVs as flat pointer-based arrays in
`TriangleMesh_C`:

```mojo
struct TriangleMesh_C(TrivialRegisterPassable):
    var points: UnsafePointer[Float32, MutAnyOrigin]   # 3 floats per vertex
    var faceIndices: UnsafePointer[Int64, MutAnyOrigin]
    var vertexIndices: UnsafePointer[Int64, MutAnyOrigin]
    var uvs: UnsafePointer[Float32, MutAnyOrigin]      # 2 floats per vertex, nullable
```

This flat layout is GPU-friendly: all mesh data lives in contiguous
host-or-device memory with no pointer-chasing through object hierarchies.

PLY meshes are loaded in `ply.mojo` and converted to the same `TriangleMesh_C`
representation.

## The Bounding Volume Hierarchy

The BVH (`bvh.mojo`) is a binary tree where each node stores an axis-aligned
bounding box. Interior nodes split their children along the axis of greatest
extent; leaf nodes contain one or more primitives. The builder uses the Surface
Area Heuristic (SAH) to choose split positions that minimize expected traversal
cost.

## Traversal

Traversal is the innermost loop of the renderer. Gonzales uses iterative
traversal with a fixed-size stack. The same Mojo function runs on both CPU
and GPU — `@parameter if has_accelerator()` dispatches to the appropriate
memory model at compile time:

```mojo
var stack = InlineArray[Int32, 32](fill=Int32(0))
var stack_ptr = 0
var node_idx = 0
while True:
    var node = nodes[node_idx]
    if bounds_intersect(node, ray, tHit):
        if node.count > 0:            # leaf
            process_leaf(node, ray)
        else:                         # interior: visit nearer child first
            var near = node.left  if dot_near else node.right
            var far  = node.right if dot_near else node.left
            stack[stack_ptr] = far
            stack_ptr += 1
            node_idx = near
            continue
    if stack_ptr == 0:
        break
    stack_ptr -= 1
    node_idx = Int(stack[stack_ptr])
```

The key optimization is **directional ordering**: the nearer child is visited
first, maximizing early exits. The fixed-size stack avoids heap allocation
per ray.

## PrimId Dispatch

Each leaf primitive is identified by a `PrimId_C` struct containing two
integer indices and a type byte. The shading kernel dispatches on the type
to select triangle intersection, sphere intersection, or area-light handling.
