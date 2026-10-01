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

## Next experiments
- Find what R4 points to: read the dispatcher with `cuda-gdb` `info cuda` / memory regions, or `compute-sanitizer`.
- Check whether a plain CUDA kernel can use `RET.ABS.NODEC` with a ray in the same register layout
  (needs cubin patching; Mojo emits PTX, so this needs a post-ptxas step).
- Capture the Vulkan ray-query shader's code the same way, if a debugger can attach to a compute queue.
- Read what Mesa NVK and envytools already document before going further.
