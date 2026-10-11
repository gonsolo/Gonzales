# NVIDIA acceleration structure format: working notes (RTX 3060, sm_86, driver 615.71.09)

Goal: build our own bottom-level structures for the RT cores, to stay small and avoid the driver's builder.
Everything here is measured on Vulkan-built BLASes; nothing is documented by NVIDIA. It may change with the driver.

## Method
- `exec/as_dump.c`: builds a BLAS from triangles in a binary file (9 floats each) and writes the driver's bytes
  (+ `.addr` with Vulkan's address and size). Builds are deterministic.
- `exec/re_sens.c`: the sensitivity map. Zeroes one 8-byte window at a time, traces a fixed ray set with the RT-core
  kernel (`build/rt_trace.cubin`) in a fresh CUDA context, and reports the windows that change the results.
  Rays must not have exactly-zero direction components (the hardware returns misses). Hit = result word 3 != 0xffffffff
  (result record per ray: t, u, v, raw R20, R8, R9).
- `exec/re_lib.py`: build/diff helpers for experiments with controlled triangles.

## Findings
1. A one-triangle BLAS is 2560 bytes, but only about 300 bytes are functional: the root offset at 0x20 (= 0x380), the
   node at 0x380, one triangle block at 0x400. The rest (header, the float table at 0x180-0x320, ...) changes the
   results in no way. Five triangles: functional windows are 0x20, 0x380-0x390, child slots 0x3a0-0x3c7,
   triangle data 0x400-0x437, and small windows at 0x468-0x47f. Vendor total for 5 triangles is 1316 bytes.
2. Header: word 0 = 0x00100003, +0x20 = offset of the root node (0x380), +0xd0 = the structure's own address.
3. Node (128 bytes at 0x380, 12 children):
   - byte 0 = 0x60 (type or flags, unchanged in all experiments);
   - bytes 1..3 = per-axis exponents e_x, e_y, e_z (extent <= 2^(e-127); 0 for a zero extent);
   - bytes 4..15 = origin p, three float32 = the minimum corner of the children's boxes;
   - 0x390..0x39f: 80 00 00 00 | 00 00 00 f0 | 00 00 00 00 | 01 00 00 00 (not decoded; byte 0x391 changed to 0x04 for 16 triangles);
   - 0x3a0..0x3ff: 12 child slots of 8 bytes: [lo_x hi_x lo_y hi_y lo_z hi_z a b]. Box coordinates are 8-bit:
     q = (x - p) / 2^(e-127) * 255, lo rounded down, hi rounded up (checked on (0.3,0.5,0.7), (2.1,0.2,0.4), (0.9,3.3,1.7):
     hi_x = 0xe6 = 230 for extent 1.8 with scale 2). An empty slot is 00 00 00 00 ff 00 00 00 (lo_z > hi_z).
     The last byte b is a per-child code (08, 10, 18, 20 and 01 seen); not decoded.
4. The whole-BLAS AABB maximum (three float32) is at 0x490 (the minimum is the node origin at 0x384).
5. Triangle block at 0x400: the first vertex as three raw float32 (0x400..0x40b); the other two vertices are bit-packed
   from 0x40c on (relative to v0 with per-axis exponents; for (0,0,0),(1,0,0),(0,1,0) word 3 is 0x1fc0007f). Not decoded.
   The block is 36 bytes for one triangle; 56 bytes for five.
6. Compaction (VK_KHR_acceleration_structure ALLOW_COMPACTION + compacted-size query + COMPACT copy) shrinks BLASes
   2.3-3x (23-33 bytes per triangle) and the TLAS to about 157 bytes per instance. A patch that adds it to
   `vulkaninterop.cpp` produced wrong renders for 3 of 4 elements (a suspected 64 KB alignment problem), so it is not in the tree.

## Next
- Decode the child code byte and the 0x390 words (mutate and watch which triangles vanish).
- Decode the triangle bit-packing (vary one vertex over many values; the shared exponent probably comes from the node).
- Find the primitive index table (windows 0x468-0x47f for 5 triangles).
- Goal: assemble our own interior nodes over Vulkan-built leaf blocks, then a full builder.

## Findings, second pass (child slots and clustering)
- For up to 12 triangles the root node is a leaf node: every used child slot is ONE triangle (its 8-bit box) and the
  last byte of the slot says where it sits in its cluster. Clusters hold up to 5 triangles and share one triangle
  block: for a cluster of m triangles the slots carry 0x08, 0x10, 0x18, 0x20 (that is (j+1) << 3 for j = 0..m-2) and
  the last one 0x01. 1 triangle: 01. 2: 08 01. 5: 08 10 18 20 01. 6: 08 10 18 20 01 | 01. 12: 5 + 5 + 2 clusters.
  Mutating child 0's code byte (values 0x00..0xff on a 5-triangle BLAS): bit 7 is ignored; 0x08 is the original;
  0x00 loses triangle 2; any other multiple of 8 loses triangle 3; any value with a non-zero low three bits leaves
  only triangle 2 visible. So bits 0..2 and bits 3..6 are separate fields; the exact semantics are open.
- Above 12 triangles the root's children are interior nodes (16 triangles: 12 slots in groups of four with codes
  08 10 20 01 | 08 18 20 01 | 10 18 20 79) and the child nodes follow at node + the word at +0x10 (0x480 for 16
  triangles, giving 0x800); for a leaf node that word is 0x80 (= 0x400 - 0x380) but values 0x88 and 0x90 also work and
  0x78 or 0 do not, so it is not a plain offset; its bit 7 must be set.
- The raw hit word is 0x80000000 + the triangle's index in the build order (one geometry); the u, v are barycentrics.
- Next: decode the interior-node codes, the +0x14 word (0xf0000000), the triangle block bit layout (36 bytes for a
  single triangle: v0 raw, then packed deltas), and how a cluster's block is located from its slot.

## Findings, third pass (triangle block coding) and what a hand-made builder could gain
- v1.x of a single triangle (v0 = 0, v2 = (0,1,0)): the low byte of the packed word at 0x40c is the IEEE exponent of the
  float for powers of two (1.0 -> 0x7f, 2.0 -> 0x80, 0.5 -> 0x7e, 4.0 -> 0x81). For x = 1 + k/64 the stored bits depend
  on the number of trailing zero bits of k: odd k gives prefix 1100 and the 4-bit payload k; k = 2 mod 4 gives 1110 and
  k/2; k = 4 mod 8 gives 1111 and k/4. This looks like a lossless variable-length float code (strip trailing zero
  mantissa bits, a prefix selects the stripped count, then the significant bits); it is consistent with the hit t, u, v
  agreeing with the driver's own to 5e-7. The full prefix table is not decoded.
- Payoff check: for 5 triangles about 150 functional bytes (about 29 per triangle) against 1316 vendor bytes. Compaction
  of large BLASes reaches 23-33 bytes per triangle, so compaction already gets to roughly the functional size. A hand-made
  builder would save build time and driver slack, not much per-triangle memory. The TLAS (157 bytes per instance
  compacted) is where a custom builder could save more, but it has not been mapped.
