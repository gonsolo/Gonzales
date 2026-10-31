# Layers

A module imports only from its own layer and the ones below it. `Tests/check_layers.py` enforces this, and
`docs/A5_library_plan.md` has the plan.

```
app          pipeline, command line, viewer
gpu          kernels and their drivers
integrators  PT, SPPM, the VCM walks, ReSTIR
paths        subpaths, vertices, connections, MIS, VCM's strategies and merge grid
estimators   generic photon stores and density estimates
transport    lobes, light sampling, walks, MNEE, SMS, texturing
scene        geometry, materials, media, lights, BVH, parsers
core         math, spectra, sampling, random numbers
```

Code that runs on both CPU and GPU is one code path, in whichever layer it belongs to. `gpu` holds only what drives it.
