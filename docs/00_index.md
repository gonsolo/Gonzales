# Gonzales — A Physically Based Renderer in Swift

## Table of Contents

0. [Introduction](00_introduction.md) — Motivation, architecture, and Swift 6.3
1. [Geometry and Transformations](01_geometry.md) — Vectors, rays, bounding boxes
2. [Spectra and Color](02_spectra_and_color.md) — RGB representation, metal optics, black-body radiation
3. [Shapes and Acceleration](03_shapes_and_acceleration.md) — BVH construction and traversal
4. [Sampling](04_sampling.md) — Piecewise distributions and Z-Sobol sequences
5. [Reflection Models](05_reflection_models.md) — BSDFs from diffuse to layered coatings
6. [Lights and Materials](06_lights_and_materials.md) — Light sources, textures, and materials
7. [Path Tracing](07_path_tracing.md) — The rendering equation in practice
8. [The Rendering Pipeline](08_rendering_pipeline.md) — Tile-based concurrency and image output

### Appendices

A. [Development Roadmap](A_development_roadmap.md) — From CPU optimization to GPU wavefront tracing

All code snippets in this book are extracted directly from the source code.
If the source changes, the book updates automatically via `make book`.
