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

## Next experiments
- Dump the stub: needs a tool that reads device code (cuda-gdb cannot); candidates are a CUPTI/SASS-patching
  tool or Nsight Graphics on the Vulkan ray-query shader.
- Check whether a plain CUDA kernel can use `RET.ABS.NODEC` with a ray in the same register layout
  (needs cubin patching; Mojo emits PTX, so this needs a post-ptxas step).
- Capture the Vulkan ray-query shader's code the same way, if a debugger can attach to a compute queue.
- Read what Mesa NVK and envytools already document before going further.
