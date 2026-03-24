# Shapes and Acceleration

A renderer spends most of its time answering one question: does this ray
hit anything? Gonzales supports several shape types — spheres, triangle
meshes, PLY files, and curves — but the key to performance is the
acceleration structure that avoids testing every shape.

## Shape Types

Triangle meshes are the dominant primitive. Each triangle stores indices
into shared vertex, normal, and UV arrays, keeping memory usage low for
meshes with millions of faces. Spheres and curves provide analytic
intersection for specific use cases (lenses, hair).

PLY meshes are loaded via a custom parser that reads the binary or ASCII
format and constructs indexed triangle arrays.

## The Bounding Volume Hierarchy

The BVH is a binary tree where each node stores an axis-aligned bounding
box. Interior nodes split their children along the axis of greatest extent;
leaf nodes contain one or more primitives. The builder uses the Surface
Area Heuristic (SAH) to choose split positions that minimize expected
traversal cost.

## Traversal

Traversal is the innermost loop of the renderer. Gonzales uses an iterative
traversal with a fixed-size stack of 32 entries:

```swift
var nodesToVisit: [32 of Int] = .init(repeating: 0)

while true {
        let node = nodes[current]
        if node.bounds.intersects(ray: ray, tHit: tHit) {
                if node.count > 0 {  // leaf
                        processor.processLeaf(node)
                } else {              // interior: visit nearer child first
                        nodesToVisit[toVisit] = farChild
                        current = nearChild
                }
        } else {
                current = nodesToVisit[--toVisit]  // pop from stack
        }
}
```

The key optimization is **directional ordering**: the nearer child is
visited first, maximizing early exits. The fixed-size stack avoids heap
allocation per ray.

## PrimId Dispatch

Each leaf primitive is identified by a `PrimId` struct containing two
integer indices and a `PrimType` enum. The scene dispatches intersection
calls based on the type — triangle, geometric primitive, transformed
primitive, or area light — keeping the BVH itself agnostic to shape details.
