# The Rendering Pipeline

This chapter describes how gonzales turns a scene description file into a
final image. The pipeline has four stages: parsing, scene construction,
tile-based rendering, and image output.

## Scene Parsing

Gonzales reads PBRT-v4 format scene files. The parser is a hand-written
recursive descent parser that handles the full PBRT-v4 syntax: transforms,
shapes, materials, textures, lights, cameras, and film settings. It builds
an intermediate representation that is then used to construct the scene
objects.

## Tile-Based Rendering

Rather than assigning individual pixels to tasks, gonzales divides the image
into rectangular tiles and renders each tile as an independent unit of work.
This improves cache locality — all rays in a tile traverse similar parts of
the BVH — and provides a natural granularity for parallel execution.

The rendering loop uses Swift's structured concurrency to dispatch tiles
across all available CPU cores:

```swift
try await withThrowingTaskGroup(of: [Sample].self) { group in
        for _ in 0..<maxConcurrent {
                guard let tile = tileIterator.next() else { break }
                group.addTask { try self.renderTile(tile: tile) }
        }
        for try await samples in group {
                allSamples.append(contentsOf: samples)
                if let tile = tileIterator.next() {
                        group.addTask { try self.renderTile(tile: tile) }
                }
        }
}
```

The loop uses **backpressure** — it seeds `maxConcurrent` tasks and adds a
new one only when a previous task completes, preventing memory spikes.

- **`withThrowingTaskGroup`**: Swift's structured concurrency ensures that if
  any tile fails, all other tiles are cancelled and the error propagates cleanly.
- **Progress reporting**: an async progress reporter runs concurrently, printing
  completion percentage without blocking render threads.

## Film and Image Output

Each tile produces an array of `Sample` values — one per pixel per sample.
After all tiles complete, the film collects these samples, applies the pixel
reconstruction filter, and writes the final image.

Image output uses OpenImageIO (OIIO) via Swift's C++ interop. This supports
writing EXR files with full HDR precision, including auxiliary channels for
albedo and normals that can be used by AI denoisers like Intel Open Image
Denoise.

## Putting It All Together

The full pipeline:

1. **Parse** the `.pbrt` file → camera, materials, shapes, lights
2. **Build** the BVH from all scene primitives
3. **Create** the light sampler with power-proportional weights
4. **Generate** tiles covering the full image bounds
5. **Render** tiles concurrently via `withThrowingTaskGroup`
6. **Write** the final EXR image via OpenImageIO

For the standard Cornell Box scene, this takes under a second on a modern
multi-core CPU. For Disney's Moana Island scene with billions of triangles
and Ptex textures, the same pipeline scales — the BVH and tile decomposition
handle the complexity without any architectural changes.
