# Embree CPU baseline for the displaced lava rock

`embree_rock.cpp` ray casts Moana's lava rock (`isLavaRocks/rockfacebg0001_geo`) with Embree 4 on the CPU, to compare
against the tessellation-free GPU prototype (`Tools/disp_proto.mojo`).

- Geometry: one `RTC_GEOMETRY_TYPE_SUBDIVISION` (Catmull-Clark, default `RTC_SUBDIVISION_MODE_SMOOTH_BOUNDARY`, which is
  OpenSubdiv's `VTX_BOUNDARY_EDGE_ONLY` as in `src/oiio/osd.cc`) built from the cage in the displacement Ptex's meta data
  (`PtexVertPositions`, `PtexFaceVertCounts`, `PtexFaceVertIndices`).
- Displacement: `rtcSetGeometryDisplacementFunction`; each grid point moves along Embree's unit `Ng` by
  `EXAG * h`, `h` from a bilinear `PtexFilter` (one per thread) at the base face `primID` and `(u, v)`. For quads
  Embree's `(u, v)` (p0 base, u to p1, v to p3) is Ptex's; the start-up probe checks this and the sign of `Ng` against
  `dP/du x dP/dv` (the GPU prototype's normal).
- Uniform `rtcSetGeometryTessellationRate` (quads per face edge).
- Camera, light and shading as the GPU tool's default view: eye = centre + diag * 1.1 * (0.45, 0.55, 0.7), 40 degree
  horizontal fov, +Y up, light = right * 0.75 + up * 0.55 - fwd * 0.25, shade = 0.08 + 0.75 * max(n.l, 0) with the hit's
  geometric normal; grey images sqrt encoded, colour images albedo (colour Ptex, gamma 2.2) * shade * 1.6, 1/2.2 encoded.
- Primary rays with `rtcIntersect1` over all hardware threads (TBB, 8 x 8 tiles). The render time is the best of
  `REPS` frames and excludes the build; the colour lookup runs after the timed pass.
- Memory: `rtcSetDeviceMemoryMonitorFunction` counts Embree's bytes (live after commit and peak during it).

## Build

    g++ -std=c++17 -O2 -march=native -o embree_rock embree_rock.cpp -lembree4 -lPtex -ltbb -lpng

Needs the Arch packages `embree` (4.x), `onetbb`, `ptex` and `libpng`.

## Run

    OUT=/path/to/images ./embree_rock

Environment (defaults in brackets): `EXAGS` [1,8,16,32], `RATES` [16,32,64,128], `WIDTHS` / `HEIGHTS` [960,960 /
540,720], `REPS` [3], `FILTER_WIDTH` [0, the finest Ptex level], `DISP` and `COLOR` (the two Ptex files),
`EMBREE_CONFIG` (e.g. `threads=12`). Images: `embree_exag<E>_rate<R>_<W>x<H>.png` and `..._color.png`.
