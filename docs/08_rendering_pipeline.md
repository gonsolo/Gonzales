# The Rendering Pipeline

This chapter describes how gonzales turns a scene description file into a
final image. Two pipelines exist — GPU wavefront and CPU tile-based — both
driven by the same `pipeline.mojo` entry points.

## Scene Parsing

Gonzales reads PBRT-v4 format scene files. The parser (`parsing.mojo`,
~2,000 lines) is a hand-written recursive descent parser that handles the
full PBRT-v4 syntax: transforms, shapes, materials, textures, lights,
cameras, and film settings. It builds C-ABI-compatible structs that are
passed directly to the render kernels without an intermediate object graph.

## GPU Wavefront Pipeline

The GPU path processes all in-flight paths at the same bounce depth in
parallel. Each frame:

1. **Generate** primary rays for all `n_pixels × spp` paths into `path_buf`
2. **Repeat** for each bounce up to `maxDepth`:
   - `traverse_gpu`: BVH traversal for all active paths
   - `shade_nee_core[use_gpu=True]`: shading, NEE, next-ray generation
3. **Accumulate** path estimates into `film_buf` and `albedo_film_buf`

The wavefront batch size is `WAVEFRONT_BATCH = 8`, meaning 8 samples are
processed together per bounce loop iteration. This amortizes kernel launch
overhead and keeps the GPU busy across all SPP.

After rendering, `mojo_gpu_atrous_denoise` runs the denoiser entirely on GPU:

1. `normalize_beauty_albedo_gpu` — divide by sample count, apply ISO scale
2. `estimate_variance_gpu` — 3×3 spatial luminance variance per pixel
3. `atrous_filter_gpu × n_passes` — variance-adaptive à-trous wavelet filter
   (Dammertz 2010), 5 passes with step sizes 1, 2, 4, 8, 16
4. Download the denoised result to host

## CPU Tile-Based Pipeline

The CPU path divides the image into 32×32 tiles and renders each tile as
an independent unit of work. Mojo's `parallelize` distributes tiles across
all available CPU cores:

```mojo
@parameter
fn render_one(tile_idx: Int):
    mojo_render_tile_v2(raster_to_camera, camera_to_world,
                        Int32(tx), Int32(ty), tx_max, ty_max,
                        sampler_params, scene, tile_buf, max_depth)
    # copy tile_buf → results at global pixel coordinates

parallelize[render_one](n_tiles)
```

After all tiles complete, `mojo_normalize_film` applies ISO scaling and
`mojo_denoise` runs the joint bilateral denoiser (guided by first-bounce
albedo). Both share data via flat `UnsafePointer[Float32]` buffers with
no heap boxing.

## Film and Image Output

Each tile produces one `TileResult_C` per pixel containing the accumulated
radiance estimate, albedo, and filter weight. The film normalizes these
by dividing by the filter weight, applies the ISO scale, and optionally
clamps to a maximum component value.

Image output uses OpenImageIO via Mojo's C FFI:

```mojo
external_call["OIIO_write_image", Int32](
    beauty_ptr, fw, fh, filename_ptr, ...)
```

This writes EXR files with full HDR precision. The albedo AOV is written
as a separate `albedo.exr` for use with external denoisers.

## Interactive Viewer

The interactive path (`mojo_render_interactive`) renders one sample per
frame and accumulates with an exponential moving average. The GPU branch
calls `mojo_gpu_atrous_denoise` with a pass count ramped by frame count
(1 pass immediately after camera movement, up to 5 passes at steady state)
to prevent the large effective radius of the full 5-pass filter from
darkening the image on the first noisy frame after a camera move.

Camera input comes from a Vulkan viewer (`viewer.mojo` → `libvulkanviewer.so`),
which writes a `CameraState` struct that `pipeline.mojo` polls each frame.

## Putting It All Together

```
Parse .pbrt → upload scene to GPU (or keep on CPU)
     ↓
GPU: wavefront bounce loop (WAVEFRONT_BATCH × maxDepth kernels per SPP batch)
CPU: parallelize over 32×32 tiles
     ↓
GPU: à-trous wavelet denoise (normalize → variance → 5 filter passes)
CPU: bilateral denoise (joint, albedo-guided, radius 7)
     ↓
Write beauty.exr + albedo.exr via OpenImageIO
```

For the Cornell Box (512×512, 64 spp), this takes ~1.1 s on a 24-core CPU
or ~0.6 s on an RTX 3060. For the Bitterli bathroom (1024×1024, 64 spp),
~30 s CPU or ~3.2 s GPU render time.
