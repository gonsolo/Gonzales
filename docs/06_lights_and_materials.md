# Lights and Materials

Light sources provide the energy that the path tracer distributes through
the scene. Materials connect surface geometry to the BSDF models described
in the previous chapter. Together they define what a scene looks like.

## Light Types

Gonzales supports four categories of light:

### Area Lights

Any shape can emit light. Area lights are the most physically accurate model
— they have finite size, produce soft shadows, and their emitted radiance
is defined per unit area. Triangle meshes tagged as emissive in the PBRT
scene file automatically become area lights.

### Point Lights

An idealized point source that emits light equally in all directions.
Point lights are delta distributions (zero area), so they cannot be hit
by random rays — they are always sampled explicitly during direct lighting.

### Distant Lights

A parallel light source infinitely far away, like sunlight. All rays from
a distant light share the same direction, which simplifies the sampling
geometry.

### Infinite Lights (Environment Maps)

An infinite area light wraps a high-dynamic-range image around the scene
as if projected onto an infinitely large sphere. This is the standard
technique for image-based lighting. The environment map is importance-sampled
using a 2D piecewise constant distribution (Chapter 4) so that bright
regions of the map receive proportionally more samples.

## Light Sampling

Choosing which light to sample is itself a sampling problem. The
`PowerLightSampler` selects lights with probability proportional to their
total emitted power, ensuring that bright lights get more attention.

For each selected light, the integrator draws a sample point on the light's
surface, computes the incident direction and radiance, and checks visibility
with a shadow ray.

## Materials

Materials are the bridge between the parser and the BSDF system. Each
material type (diffuse, conductor, dielectric, coated conductor, mix, etc.)
reads its parameters — reflectance colors, roughness values, refractive
indices — and constructs the appropriate `FramedBsdf` at each surface
interaction point.

Gonzales uses a `BsdfVariant` enum for type-erased BSDF dispatch, avoiding
existential overhead while keeping the material system extensible.

## Textures

Surface parameters like color and roughness can vary across a surface.
Gonzales supports two texture types:

- **Image textures** — loaded from PNG, EXR, or other image formats via
  OpenImageIO, with bilinear filtering and UV mapping
- **Ptex textures** — Disney's per-face texture format, accessed via the
  Ptex C++ library through Mojo's C FFI. Ptex eliminates UV mapping
  entirely — each mesh face stores its own texture data directly.
- **Shader programs** — a material node graph compiled to a flat program
  (`scene/appearance/shader.mojo`), in the spirit of Cycles' SVM: Int32
  instructions `[op, dst, a, b, c, d, i0, i1]` on sixteen four-float
  registers, covering Cycles' math, vector math, mix and blend modes, colour
  ramp, curves, hue/saturation, map range, mapping, image and normal-map
  nodes. A program is an entry of the texture table (its words hex-encoded
  in the entry's name, `#gzs:...`), so it plugs into any texture slot a
  material already has — base colour (`tex_idx`), roughness
  (`rough_tex_idx`), emission (`emit_tex_idx`), normal map and bump — and
  every integrator that samples that slot runs it at the hit's UV.
  The Blender add-on compiles node trees with `src/hdgonzales/blender/shader.py`
  and hands them over through `gz_scene_set_shader`; a graph it cannot
  compile falls back to the baked value. Outputs are RGB and go through the
  same RGB-to-spectrum upsampling as image textures.
