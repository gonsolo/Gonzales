# Layers

A module imports only from its own layer and the ones below it. `Tests/check_layers.py` enforces this, and
`docs/A5_library_plan.md` has the plan.

```
app          pipeline, command line, viewer
cpu, gpu     how a render is driven, per integrator; peers: threads and tiles, kernels and buffers
integrators  one path step, shared by cpu and gpu: pt/, vcm/, sppm/
paths/vcm    subpaths, vertices, connections, MIS, strategies and the merge grid
estimators   generic photon stores and density estimates
transport    lobes, light sampling, walks, MNEE, SMS, texturing
scene/description  parsers, scene building
scene/accel        BVHs and ray-tracing backends
scene/appearance   materials, textures, media, lights
scene/shapes       primitives, curves, mesh files
core         math, spectra, sampling, random numbers
```

Code that runs on both CPU and GPU is one code path, in whichever layer it belongs to. `cpu` and `gpu` hold only what drives it.
