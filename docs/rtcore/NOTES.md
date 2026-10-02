# RT core lab notes (RTX 3060, sm_86, driver 615.71.09, OptiX SDK in /opt/optix)

Goal: learn how the hardware ray-tracing unit is driven, to see whether CUDA/Mojo kernels can reach it
without Vulkan or OptiX at trace time.

## Reproduce
    cd /tmp && cmake /opt/optix/SDK -DOptiX_INSTALL_DIR=/opt/optix && make optixTriangle
    ncu --set basic --import-source yes --export tri ./bin/optixTriangle -f out.ppm
    ncu -i tri.ncu-repz --page source --csv --print-source sass          # user shader pieces
    cuda-gdb -batch -x cmds ./bin/optixTriangle   # set cuda break_on_launch application; run; disassemble $pc,$pc+0x200

## Findings (2026-10-02)
1. `optixLaunch` is not one kernel. The driver compiles each shader into separate entry pieces
   (`__raygen__..._ss_0`, `_ss_1`, `__closesthit__...`, `__miss__...`); `ss_N` are continuation pieces
   split at each `optixTrace`.
2. The raygen piece before the trace contains no RT-specific mnemonic. It only fills registers and ends
   with `RET.ABS.NODEC R4, 0x0`:
   - R10-R12 origin, R14-R16 direction, R9 tmin, R17 tmax (0x5a0e1bca = 1e16),
   - R18:R19 = traversable handle, loaded from constant bank 3,
   - R20 = 0, R21 = 0xffffffff (continuation id), R37 = 0xff01 (visibility mask / flags).
   R4 holds the address to return to (0xc268450 here). The code at that address could not be read in
   cuda-gdb (`x/i` returns empty), so the traversal dispatcher is not visible to us.
3. The closest-hit piece reads its hit data from memory (`LD.E.STRONG.SM R108,[R2]` through
   c[0x3][0x40]) and also ends in `RET.ABS.NODEC R4`. The hit record is therefore written by the
   hardware/dispatcher, not by the shader.
4. `libnvoptix.so` (sm_86 cubins, disassembled with nvdisasm) and the on-disk OptiX cache
   (`/var/tmp/OptixCache_$USER/optix7cache.db`, cubins reassembled with sqlite3, not by scanning raw
   bytes: blobs span database pages) contain no RT-specific mnemonic either. The cache holds
   the compiled runtime pieces with ordinary FFMA/FMUL/BSSY/BRX code.
5. Hypothesis: the trace is a hand-off through the special return, with the ray in a fixed register
   window, to an SM-internal dispatcher that drives the RT core. To test next.

## Follow-up (same day)
- R5:R4 = 0x100_0c268450 at the trace RET, 0x10250 bytes above `ss_0` (0x1000c258200) and not equal to
  `ss_1` (0x1000c259000). It is not a continuation address. `x/xw` on it and even on the known raygen code
  returns zeros, i.e. cuda-gdb cannot read device code through plain memory reads, and `disassemble`
  only works for ELF-symbolized functions. The target is an unsymbolized stub, so it stays invisible here.
- Prior art (web search): Mesa NVK documents that Vulkan ray tracing on NVIDIA lacks information on
  how the shader side works; there are public SASS references for other instructions (e.g. sass-king,
  nvidia-sass-document) but none for the RT hand-off. Nothing to reuse.

## Dead ends (2026-10-02)
- CUPTI injection (`CUDA_INJECTION64_PATH`) sees only context/stream/memory callbacks from OptiX;
  no `MODULE_LOADED` and no `cuModule*`/`cuLink*` driver calls. OptiX loads its modules through an internal
  driver path, so the public API cannot dump them.
- User-triggered GPU core dump (`CUDA_ENABLE_USER_TRIGGERED_COREDUMP`, `CUDA_COREDUMP_PIPE`) did not create
  the trigger pipe in this setup.
- Exception-triggered core dump: tried with a patched sample (bad output pointer, illegal address in
  the raygen continuation). The driver prints "Starting GPU coredump generation" and then writes no file,
  with or without `skip_abort`, with default and explicit `CUDA_COREDUMP_FILE`. OptiX contexts apparently
  cannot be dumped this way. (The original text of this item:) exception-triggered core dump (`CUDA_ENABLE_COREDUMP_ON_EXCEPTION`) from a deliberately faulting
  launch, then `target cudacore` in cuda-gdb; set `CUDA_COREDUMP_GENERATION_FLAGS=skip_global_memory,...` to
  keep the file small (an earlier session filled the root disk with core dumps).

## Prior art (web search, 2026-10-02)
- NVIDIA forum answers: RT cores are reachable only through OptiX, DXR and Vulkan; no PTX instruction
  for them is public, and NVIDIA says the interface changes every generation.
- The only public description of the hardware is patents on the Tree Traversal Unit (TTU), e.g. US11928772B2.
- Mesa NVK has no ray-tracing support yet and lists missing shader-side information.
- No public reverse-engineering of the SM-to-RT-core hand-off found.

## Nsight Graphics (installed from AUR, 2026.3.1)
- Headless GPU Trace works: `ngfx --activity "GPU Trace Profiler" --exe ./trace_loop --args 40
  --start-after-submits 5 --limit-to-submits 3 --auto-export --output-dir out` (do NOT pass `--platform`,
  Qt swallows it). `trace_loop.c` (this directory) drives the Vulkan ray-query backend.
- The trace's Shader Pipelines table lists our compute shader plus driver-internal "Ray Tracing Internal
  Lo..., Acceleration..., Scheduler, Traversal, Geometry..." shaders, so the Vulkan ray-query path runs through
  driver-generated scheduler/traversal code. All of them show N/A source.
- No SASS: the shader profiler recorded 0 samples ("Missing Metrics Data") from both the command line and the
  GUI; the Shader Source tab offers only the SPIR-V of our shader. Window screenshots work with
  `DISPLAY=:0 import -window <id>` (XWayland); gnome-screenshot is blocked, xdotool clicks do not register.

## The ray-query SASS (found 2026-10-02)
Nsight Graphics shows no SASS, but the driver reports it through two other channels:
- `VK_KHR_pipeline_executable_properties` (supported, `pipelineExecutableInfo = true`) gives statistics for
  our `intersect_batch.comp`: 56 registers, "Binary Size" 5376 bytes (= 336 SASS instructions), no stack,
  no shared memory. It exposes no internal representation (SASS text).
- `vkGetPipelineCacheData` returns the compiled binary: zstd frame at byte 0x64; decompressed blob starts
  `NVDANVVMNVuc`; machine code at 0x178, `Binary Size` bytes long. `pipeline_dump.c` + `extract_sass.py`
  here dump it and patch it into a dummy sm_86 cubin so `nvdisasm` can print it (`ray_query_shader.sass`).
- nvdisasm decodes 320 of the 336 instructions and silently OMITS the other 16. They are one contiguous block
  at 0x6d0-0x7c0, right after the shader moves the ray into R8-R15 (origin, tmin, direction, tmax) and
  loads constants into R20-R24. The undecoded opcodes (12-bit field in bytes 0-1):
  `0x3d0` x1, `0x9d4` x1, `0x3d1` x8, `0x3d3` x1, `0x3d2` x5.
  These are the ray-query / RT-core instructions: everything else in the shader is ordinary SASS (IMAD, LDG,
  ISETP, ...), and after the block the code tests bit fields of R23 (0xfe000000 / 0x3f) as a status word.
- The raw bytes (0x6d0-0x7c0) are in `extract_sass.py`'s output; first words: 3d0 `d0730000 00000000 00010000
  00ec0f00`, 9d4 `d4790000 000e0000 00000000 00e80f02`, 3d1 `d1730000 16000000 14000000 00e20100`.

## Variant diffs (2026-10-02, `variants/run_variants.py`)
`nvdisasm --binary SM86 code.bin` reads the raw code directly (no dummy cubin needed). Each variant of
`intersect_batch.comp` was compiled by the driver and the undecoded block compared with the base shader:
- The block is always the same shape: `0x3d0` (constant bytes `d0730000 00000000 00010000 00ec0f00`),
  `0x9d4`, eight `0x3d1`, one `0x3d3` (constant), then a run of `0x3d2`.
- Ray flags (opaque / terminate-on-first-hit / no flags / cull back faces) and the cull mask (0xff, 0x01,
  dynamic) do NOT change the block's bytes (only its position, because the code before it changes). They are
  set up in registers before the block (`IMAD.MOV.U32 R20/R22/R24, ... immediate`; e.g. R20 = -0xfcfdf9,
  R24 = -0xfcfbf9, R22 = 0xff0000 in the base shader), so the flags and mask travel as register values.
- The ray itself is moved into R8-R15 (origin, tmin, direction, tmax) and R16-R23 just before the block.
- `tmin` constant 0.0 only flips a scheduling byte of `0x9d4` (`02` -> `00`).
- The number and registers of the `0x3d2` instructions follow the result queries: base (type, barycentrics,
  t, custom index, primitive index) has five, a variant that drops t / barycentrics / custom index / primitive
  index has four. Their byte 2 is a destination register in steps of 4 (R8, R12, R16, ...), byte 8 a source
  register (R10, R14, R18, RZ, R22), i.e. they read results out of the unit; the eight `0x3d1` take register
  pairs and look like writing the ray into the unit (byte 4 and byte 8 are register numbers that change with
  register allocation). These readings are guesses from operand patterns, not confirmed.

## Running the driver's ray-query code as a CUDA kernel (2026-10-02, `exec/`)
Pipeline: `as_probe` builds a 32-triangle mesh with the Vulkan backend and saves the BLAS/TLAS bytes (via the new
`vulkanrt_debug_read_as`), 256 rays and Vulkan's own hit results; `make_cubin.py` wraps the driver's 5376 code bytes
as kernel `k` of an sm_86 cubin (skeleton from a dummy CUDA kernel with 72 live floats so the register count is
large enough; with too few registers the launch fails with ILLEGAL_INSTRUCTION) and rewrites the constant-bank
operands (Vulkan's c[0][0x20..0x58] and c[1][0], c[1][8]) to kernel parameters at c[0][0x160..0x198];
`run_rt2` relocates the structures into one CUDA allocation and launches.
- The shader's inputs in constant bank 0: 0x20 rayCount, 0x30/0x34 TLAS address, 0x40 rays pointer, 0x48 rays
  buffer size, 0x50 results pointer, 0x58 results size; bank 1 offsets 0 and 8: two 64-bit values used as bases in
  the post-hit decode (stride 0x480).
- Each acceleration structure stores its own absolute virtual address at +0xd0, and the TLAS stores the BLAS address
  at +0x200. CUDA cannot be given Vulkan's addresses (fixed-address reservation is refused), so the run patches those
  three words. Nothing else needed relocation.
- RESULTS: the 16 RT instructions EXECUTE in a CUDA kernel. No ILLEGAL_INSTRUCTION. With the BLAS as root the launch
  succeeds and every ray misses (Vulkan: 239 of 256 hit). With the TLAS as root the warp dies with
  "Warp MMU Fault at PC 0x0": the unit transferred control to address 0, i.e. it expects a handler (the driver's
  "Ray Tracing Scheduler/Traversal" shaders that Nsight lists) that a CUDA context has not installed.
- Register state around the block (thread 0, BLAS root): R8:R9 = root node address with bit 60 set
  (`0x1000b802380 | 1<<60`), R20/R22/R24 = flag words (0xff030207, 0xff0000, 0xff030407); after the block R20-R23 =
  0xffffffff and R8-R19 = 0 (no hit). Changing the pointer tag nibble (all 16 values) or the BLAS alignment (8 KB to
  1 MB) or adding 4-64 KB of per-thread local memory made no difference.
- Open: what else the Vulkan context provides (per-context RT state, a handler address for instance traversal, the
  two bank-1 values). A debugger breakpoint at the block entry in a real Vulkan dispatch is not possible with the tools
  here, so the next step would be to find where the TLAS-miss PC=0 comes from (the driver's internal shaders).

## Single-stepping the block in cuda-gdb (thread 0, BLAS root, CUDA context)
`tbreak *($K+0x6d0)` then `stepi` with all of R0-R47 dumped after each step:
- `0x3d0` (at 0x6d0): no register changes.
- `0x9d4` (at 0x6e0): R8:R9 (root node address) become 0, R20-R23 become 0xffffffff, and the program counter goes
  straight to 0x7d0 -- the `0x3d1`/`0x3d3`/`0x3d2` instructions at 0x6f0-0x7c0 are never executed. So `0x9d4` is
  the gate of the query: it takes the root pointer and the flag words, and on failure it writes -1 to R20-R23 and
  branches over the setup/readout instructions. In the Vulkan run it must succeed and fall through.
- After the jump the shader tests R23 (0xfe000000 / 0x3f bit fields) and writes the miss result.
Therefore nothing in the ray-reading `0x3d1`/`0x3d2` instructions has been observed yet. Why `0x9d4` fails in a CUDA
kernel is unknown. Compared with a plain compute shader, the pipeline blob of the ray-query shader has extra header
records (tags 0x10, 0x17 and a 0x22/0x20 pair, record count 3 instead of 1) that probably describe the RT resources
the driver sets up per launch; the OptiX cache cubins contain none of the RT block, so no extra attribute could be
compared.

## RESULT: hardware ray tracing from a plain CUDA kernel (2026-10-02)
Perturbing the inputs of the gate instruction `0x9d4` at 0x6e0 (cuda-gdb: `set $R21=0` before `stepi`, `gate.py` in
`exec/`) showed it is the TRACE instruction, not a gate:
- Inputs: R8:R9 = root node address (BLAS address + the offset stored at AS+0x20; the shader ORs 1<<60 into the
  high word), R20-R23 = ray/flag words, and the ray itself, loaded earlier by `0x3d0` from R0-R7 (origin.xyz, tmin,
  direction.xyz, tmax; the operands of `0x3d0` are constant).
- Outputs: R20 = hit record (`0x8000001b`: top bits are a hit/leaf kind, the low bits the primitive index; misses give
  `0xffffffff`), R21 = t, R22 = u, R23 = v (IEEE floats). R8:R9 are consumed (zeroed).
- With the driver's value R21 = 0x455420 the unit refuses the query in a CUDA context (all outputs -1). R21 = 0 makes it
  trace. (R21 is computed from the ray direction by the preceding compare/OR chain; what its bits mean is unknown.)
- After the instruction control jumps to 0x7d0; the instructions 0x6f0-0x7c0 (`0x3d1`, `0x3d3`, `0x3d2`) are not
  executed on this path. Their role is still unknown (not needed for closest-hit on a BLAS).
- The driver's own post-processing (instance/primitive decode through two tables in constant bank 1) faults in CUDA
  because those tables do not exist, so `exec/make_trace_kernel.py` replaces everything after the trace with stores
  of R21, R22, R23, R20 (record = t, u, v, raw R20).
- Verified against Vulkan's ray-query shader on the same rays: 32 triangles / 256 rays (239 hits) and 8192
  triangles / 1,048,576 rays (1,044,490 hits): hit/miss, t, u, v (max abs error 5e-7) and triangle index
  (`R20 & 0x1fffffff`) all agree for every ray. Timing, 1M rays on the 8192-triangle mesh: 0.259 ms per launch
  = about 4.1 Grays/s on the RTX 3060 (rays are random start points on a plane, not a representative workload;
  no software-BVH baseline measured yet).
- Reproduce: `as_probe <dir> <grid> <rays>` builds the BLAS with Vulkan and writes bytes + reference hits;
  `pipeline_dump` + `variants/run_variants.py`-style extraction gives `code.bin` (5376 bytes);
  `make_trace_kernel.py code.bin code_trace.bin`; `make_cubin.py code_trace.bin x.cubin 56`;
  `BLAS_OFF=0x40000 REPS=50 run_rt2 x.cubin <dir> blas`.
- Limits: TLAS roots fault ("Warp MMU Fault at PC 0x0", the unit seems to call a driver-installed handler for
  instances), so only a single BLAS can be traced this way; any-hit/intersection programs, curves, instancing are untested.

## Software baseline (exec/sw_bvh_baseline.cu, same mesh, same 1M rays, RTX 3060)
CPU-built binary BVH (median split, <=4 triangles/leaf), GPU stack traversal, Moller-Trumbore; validated against
Vulkan's hits (0 mismatches). Plain closest-hit, opaque triangles, rays start on a plane above a wavy grid:
- 8,192 triangles:   software 557 Mrays/s;  RT hardware via the CUDA kernel 4,056 Mrays/s  (7.3x)
- 131,072 triangles: software 258 Mrays/s;  RT hardware via the CUDA kernel 1,488 Mrays/s  (5.8x)
(the 131k-triangle RT run also matches Vulkan on all 1,048,576 rays). A simple median-split software BVH is not
Gonzales' own traversal; it is a lower bound on what software can do on this workload, not a measurement of the
renderer.

## Library and Mojo binding (2026-10-02)
- `src/rtcore/` (`librtcore.so`, CUDA driver API, stub without CUDA): `rtcore_create(cubin, as_bytes, size, vk_address)`
  loads the cubin into the current CUDA context, copies the acceleration structure and relocates its stored addresses;
  `rtcore_trace(handle, rays_dptr, results_dptr, n, stream)` enqueues the trace. Linked into the Gonzales binary.
- `make rt-cubin` (or `docs/rtcore/exec/build_rt_cubin.py`) generates `build/rt_trace.cubin` from the installed
  driver: Vulkan compiles `intersect_batch.comp`, the code is read from the pipeline cache and patched. The patcher
  checks the six instructions it rewrites and refuses a driver that compiled the shader differently. Nothing from
  NVIDIA is committed.
- `src/gonzales/rtcore.mojo` and `Tests/unit/test_rtcore.mojo`: Vulkan builds a BLAS, Mojo reads its bytes
  (`vulkanrt_debug_read_as`), traces 4096 rays on its own `DeviceContext` stream, and compares every ray with Vulkan's
  ray query (3431 hits, 0 mismatches). The test skips without the cubin or without CUDA.

## TLAS / instances (2026-10-02)
- The earlier "Warp MMU Fault at PC 0x0" for a TLAS root was NOT a missing driver handler. The TLAS stores each BLAS
  address twice: as the full 64-bit address at +0x200 and, in the instance leaf node, as `address >> 16` in a 32-bit
  field at the unaligned offset 0x60a. Only the first was relocated, so the unit followed a stale address (stepping the
  trace instruction showed it returning the old Vulkan address `0xdf3dfc0380` in R20:R21). Patching the shifted field
  too fixes it. BLAS addresses must therefore be 64 KB aligned (Vulkan gives 128 KB). Creating an OptiX context on
  the CUDA context first made no difference.
- With that, a TLAS over three BLASes (`as_probe <dir> 4 1024 3`, `exec/run_rt4.c`): all 1024 rays match Vulkan on
  hit/miss, t, u, v and the per-BLAS triangle index (`R20 & 0x1fffffff`).
- The instance (mesh) index is NOT in the registers the kernel stores: R8:R9 come back zero, R20's top bits are the same
  for all instances (0x8 in the high nibble here), and instanceCustomIndex does not appear. The Vulkan shader reads it in
  its post-processing through tables in constant bank 1 (stride 0x480 from an address that includes the TLAS base), so
  decoding that path is the open problem. Until then, trace ONE merged triangle geometry (one BLAS, global triangle
  index) and map the index back to (mesh, triangle) on our side; that needs no instance information.

## Wired into the renderer: `--rt-hardware` (2026-10-02)
`gonzales --gpu --rt-hardware scene.pbrt` (wavefront GPU path tracing; implies `--vulkan-rt-shade`; set
`GONZALES_RTCORE_CUBIN` if you do not run from the repository root). Triangle-only scenes without instancing, curves
or spheres; otherwise it prints a note and uses the Vulkan path. How it works: every mesh is merged into ONE geometry,
Vulkan builds its BLAS, `librtcore` traces it from a CUDA kernel into the interop rays/results buffers, and
`rt_convert.cu` turns the raw records into the interop Result layout (mesh and local triangle from prefix sums of the
per-mesh triangle counts), so `vulkaninterop_unpack_results_kernel` and everything after it are unchanged. Not done: the
VCM wavefront path (`--vcm --vcm-wavefront`), shadow rays (they still use the software BVH), any-hit/alpha inside the trace.

Measured, 640x360, RTX 3060, seed 1, per-sample cost from the slope between 64 and 256 spp (the totals include setup):

| scene (triangles)        | software BVH | Vulkan ray query | RT hardware (CUDA) |
|--------------------------|--------------|------------------|--------------------|
| bistro_vespa (2.83M)     | 80.9 ms/spp  | 66.7 ms/spp      | 62.6 ms/spp        |
| ganesha (4.32M)          | 21.1 ms/spp  | 21.3 ms/spp      | 20.8 ms/spp        |

Setup (seconds, same runs): bistro 9.3 / 13.4 / 17.2, ganesha 2.6 / 3.8 / 8.4. The hardware path builds a second,
merged acceleration structure and merges the meshes in a single-threaded Mojo loop. Images: means agree to 3 digits;
RT hardware vs Vulkan 0.24% (ganesha) and 3% (bistro) relative RMS per pixel at 32 spp, software vs either about 1% and
23% (floating-point differences in the ray tests change individual path decisions; the means agree, so not bias).
The ray-casting speed-up measured in isolation (5.8-7.3x) shrinks to 1.0-1.3x for the whole path tracer because
shading, texture lookups and the pack/convert/unpack kernels dominate.

## Prior art found by web search (2026-10-02)
- NVIDIA caches compiled shaders (Vulkan and OpenGL) in `~/.nv/GLCache` or `~/.cache/nvidia/GLCache`;
  `nvcachetools` (reads `.toc`/`.bin`) and `nvucdump` (extracts sections of `.nvuc` objects) plus `nvdisasm --binary SMxx`
  or `envydis` are the known extraction route: https://danilw.github.io/blog/decompiling_and_optimizing_nvidia_shaders/
  That article does not cover ray tracing.
- SASS references: cicuvc/nvidia-sass-document, florianmattana/sass-king, cloudcores/CuAssembler, NoxNode/AmpItUp
  (Ampere encoding). None documents the RT instructions.
- Patents describe the Tree Traversal Unit (TTU) hardware. No public reverse-engineering of the SM to RT-core
  interface was found.

## Next experiments
- Dump the stub: needs a tool that reads device code (cuda-gdb cannot); candidates are a CUPTI/SASS-patching
  tool or Nsight Graphics on the Vulkan ray-query shader.
- Decode the 16 opcodes: vary the shader (ray flags, tmin/tmax use, any-hit vs closest, committed query types,
  ray count in flight) and diff the blocks; then try to emit them from a patched CUDA cubin against a
  Vulkan-built AS (interop memory).
- Check whether a plain CUDA kernel can use `RET.ABS.NODEC` with a ray in the same register layout
  (needs cubin patching; Mojo emits PTX, so this needs a post-ptxas step).
- Capture the Vulkan ray-query shader's code the same way, if a debugger can attach to a compute queue.
- Read what Mesa NVK and envytools already document before going further.

## Pseudo-assembler listing (trace_kernel.sasm)

The instructions we add to the driver's compiled shader are now written as pseudo SASS in `docs/rtcore/exec/trace_kernel.sasm`
(with the trace instruction 0x9d4 documented in its header). `make_trace_kernel.py` assembles that file into 16-byte
words (templates cloned from instructions the driver emitted) and verifies the result by disassembling it with nvdisasm.
The resulting cubin is byte-identical to the one the earlier hand-patching produced.

Setup cost: the AS readback used an uncached staging buffer (3.5 s for 295 MB); with HOST_CACHED it is 0.07 s, and the
interop scene is now built from the merged geometry so only one acceleration structure is built (ganesha setup 8.4 s -> 4.8 s).


## Shadow rays on the RT cores

The shade kernels used to trace NEE shadow rays inline with the software BVH. With `--rt-hardware` they now defer them
into `SHADOW_SLOTS` (2) slots per path (`_shadow_contribute`; a candidate that finds no free slot, a scene with guiding,
or a scene with alpha cutouts stays inline). After the last material kernel, `rtcore_shadow_rays_gpu` runs one pass per
usable slot: pack the slot's rays, trace them on the RT cores (closest hit; a miss means the light is visible), add the
contribution of the unobstructed ones. A second slot is only used when the scene has two kinds of light (it costs about
13 ms/spp otherwise, since most of its rays are dead). Memory: the task buffer is only allocated with the slots when
`GONZALES_RTCORE` is set. `GONZALES_RTCORE_NOSHADOW=1` turns the shadow rays off again (primary rays only).

Bug found on the way: the hardware returns `u` with a negative sign on some hits (same magnitude; Vulkan reports it
positive). The convert kernel now takes `fabsf(u)`. Before the fix every hardware render of a textured/smooth-shaded scene
was ~0.9% too dark (living-room, 512 spp: 0.9907, after: 0.9999). Found by tracing the same rays through the Vulkan
ray query and the RT-core kernel on the same acceleration structure and comparing u, v, t per ray.

ms per sample, 640x360, GPU path tracer (software BVH / Vulkan ray query / RT cores primary only / RT cores + shadows):

| scene | software | Vulkan | RT primary | RT + shadows |
|---|---|---|---|---|
| living-room | 69 | 79 | 80 | 80 |
| staircase | 61 | 83 | 80 | 82 |
| dining-room | 83 | 75 | 74 | 27 |
| kitchen | 75 | 72 | 74 | 62 |
| ganesha | - | 15.8 | - | 15.8 |

Bistro has alpha cutouts, so its shadow rays stay inline (hardware primary only: 60 vs 74 ms software). The hardware
shadow rays win where the software BVH is slow (dining-room 3x) and are neutral elsewhere; the small scenes lose to the
software BVH because of the extra pack / trace / resolve passes. Means agree with the software renderer within 0.1%.


## Instancing and spheres (2026-10-02)

**Instances without flattening.** A trace against a TLAS stores the hit instance in register R42 as `instance index + 1`
(0 on a miss); found by dumping registers 0..47 after the trace for a TLAS over three BLASes (`exec/dump_regs.py`: R42 is
1/2/3 for rays that hit mesh 0/1/2). The trace kernel now stores it in record word 4 (`trace_kernel.sasm`). The instance
index is the position in the TLAS instance array: the ordinary (non-template) meshes in index order, then the object
instances (vulkaninterop_rt_create_scene).

**Hit word of a BLAS with several geometries** (a template with one geometry per mesh): bit 31 clear, `k = word >> 29`,
`s = 28 - 4k`; the geometry index is in bits 28..s and the triangle number within that geometry in the low s bits
(9-geometry template: word >> 24 = 0x20 + geometry, low 24 bits = triangle). A single-geometry BLAS has bit 31 set and the
triangle in the low 29 bits. Verified against Vulkan's decoded mesh/triangle/geometry on every ray of the barcelona
pavilion at night (43 instances of 2 templates, 107 meshes): 0 mismatches over 6 bounces.

`librtcore` now takes several acceleration structures (`rtcore_create_scene`: TLAS first, each structure at a 64 KB aligned
offset, 8-byte and shifted 32-bit references relocated) and per-instance decode tables (`rtcore_set_domains`). Instanced
scenes use the Vulkan scene as built (TLAS over per-mesh BLASes plus one multi-geometry BLAS per template); scenes without
instancing keep the single merged BLAS, which is faster to trace.

**Spheres** stay analytic: the primary rays run the existing sphere pass after the unpack, and the hardware shadow resolve
tests the spheres per deferred ray. Only curves are still excluded.

Pavilion at night (path tracer, 640x340, ms per sample): software 89, Vulkan ray query 81, RT cores 87 (shadow rays stay on
the software BVH because the scene has alpha cutouts); setup 11 s against 5 s.


## Alpha cutouts (2026-10-02)

The acceleration structure stays opaque (the hardware trace has no any-hit programs on this path), so a ray returns the
nearest triangle even when its alpha cutout rejects it. `rtcore_alpha_passes` fixes that in software around the hardware
trace: after each trace `rtcore_alpha_kernel` applies `alpha_killed` (the test the software BVH runs) to every pending ray's
hit; a rejected hit moves the ray's t_min just past it (`t * 1.00001 + 1e-5`) and the ray is traced again, anything else is
final and gets tmax = 0 so later traces skip it. Up to `RT_ALPHA_PASSES` (4) traces per call; a ray that still has a
rejected hit after the last one (dense foliage: a Bistro hedge has more layers than any fixed count) is flagged (hit flag 3)
and resolved by the software BVH, which has no depth limit (`rtcore_alpha_fallback_gpu` for primary rays, the shadow resolve
kernel for shadow rays). Shadow rays use the same passes. Scenes without alpha are untouched.

Bistro vespa, 640x360, ms per sample: software 74, Vulkan 52 (it ignores alpha), RT cores with alpha 35; mean
1.0009 of the software image. With 3 passes and no fallback the image was 16% too bright (rays leaving the foliage
as misses); 12 passes still 6%. Synthetic check: the Cornell box with a half-transparent and a fully transparent quad,
hardware / software = 1.00003. Pavilion at night: 65 ms (software 89). Bathroom: 1.0007.


## VCM on the RT cores (2026-10-02)

`gonzales --gpu --vcm --rt-hardware scene.pbrt` runs the staged VCM driver (`--vcm-wavefront`, implied) with the RT-core
trace for the light paths, the camera paths and the connection shadow rays. The staged driver renders pixel-identical
images to the megakernel `vcm_render_gpu` (checked on the pavilion: ratio 1.0, rms 0), so the paper's thinning, footprint
and keep-aware MIS are all there. Instancing (TLAS), spheres (analytic pass from the rays buffer; the shadow resolve sends
rays that hit a sphere to the full visibility trace) and alpha cutouts (same re-trace passes; deep tail to the software BVH)
are supported; curves are not. The shadow resolve decodes instance hits through the instance table and treats a ray with
too many rejected alpha hits as needing the full trace.

Correctness: Cornell box 0.99999 of the software staged VCM; pavilion at night (43 instances, 2 spheres, alpha) 1.0001.

Speed: VCM is not trace bound (merging and connecting dominate), so there is no gain. Pavilion at night, 640x340, ms per
sample: staged software 854, RT cores 948; without the alpha passes 841 (the passes re-trace the 10-slot shadow batch up to
four times, mostly dead rays). Setup 10.6 s against 4.5 s. The staged driver keeps per-path state for every pixel and runs
out of memory at 1920x1080 on a 12 GB card, with or without the RT cores (the megakernel does not), so the hardware teaser is
rendered at 1280x720: 128 spp in 442 s.
