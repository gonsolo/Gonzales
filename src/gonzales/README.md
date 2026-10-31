# Layers

A module imports only from its own layer and the ones below it. `Tests/check_layers.py` enforces this, and
`docs/A5_library_plan.md` has the plan.

```
app          pipeline, command line, viewer
cpu, gpu     how a render is driven: threads and tiles, kernels and buffers; peers
integrators  one path step of PT, SPPM, VCM, ReSTIR, shared by cpu and gpu
paths        subpaths, vertices, connections, MIS, VCM's strategies and merge grid
estimators   generic photon stores and density estimates
transport    lobes, light sampling, walks, MNEE, SMS, texturing
scene        geometry, materials, media, lights, BVH, parsers
core         math, spectra, sampling, random numbers
```

Code that runs on both CPU and GPU is one code path, in whichever layer it belongs to. `cpu` and `gpu` hold only what drives it.
