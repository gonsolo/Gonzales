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

Traversal is the innermost loop of the renderer. Gonzales uses a stackless
iterative traversal with a fixed-size array of 32 entries — enough for any
reasonable tree depth, and avoids heap allocation entirely:

{{snippet:Sources/libgonzales/Accelerators/BoundingHierarchy.swift:bvh-traverse}}

Key performance details:

- **Directional ordering**: the traversal visits the child that is closer
  to the ray origin first. This maximizes early exits when a closer
  intersection has already been found.
- **Fixed-size stack**: `[32 of Int]` uses Swift's inline storage for
  value-type arrays, avoiding any heap allocation per ray.
- **Generic leaf processing**: the `LeafProcessor` protocol lets the same
  traversal code handle both occlusion queries (shadow rays) and full
  intersection queries (camera rays) without code duplication.

## PrimId Dispatch

Each leaf primitive is identified by a `PrimId` struct containing two
integer indices and a `PrimType` enum. The scene dispatches intersection
calls based on the type — triangle, geometric primitive, transformed
primitive, or area light — keeping the BVH itself agnostic to shape details.
