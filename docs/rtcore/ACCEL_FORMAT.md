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

## Findings, fourth pass (the triangle block is decoded; vendor builder used only as an oracle)
Tools: `exec/re_oracle.py` reads back what the hardware decodes for ANY bytes (trace a ray grid, solve the hit barycentrics
for the three vertices; patch bits and watch), `exec/re_codec.py` encodes a one-triangle block.
- A triangle block is 128 bytes at node+0x80 (0x400 for a one-leaf BLAS): v0 as three raw float32 (0x00..0x0b), a bit
  stream from 0x0c, index/prim words at 0x68..0x77, the primitive word at 0x78 (0x80000000 + index of the block's first
  triangle) and the DESCRIPTOR at 0x7c. The descriptor, not the node, says how the stream is parsed.
- Descriptor (one triangle): bits 0..14 = three 5-bit fields, (width_x - 1), (width_y - 1), (width_z - 1); bits 24..31 =
  55 - m; bits 20..23 = (triangles in the block) - 1. m is the shared mantissa length: the largest number of significant
  mantissa bits over ALL vertices including v0 (the lower mantissa bits of a stored value come from v0, so v0 must be
  counted). width_axis = m + e_axis, e_axis = bit length of the xor of the 9-bit sign+exponent fields (float bits 23..31)
  of every stored vertex coordinate with v0's, at least 1.
- Stream (one triangle): v1.x v1.y v1.z v2.x v2.y v2.z, each field LSB first: the top m bits of the mantissa (bits
  22..23-m), then the low e_axis bits of sign+exponent; the high bits of sign+exponent are v0's. Verified on the hardware
  for 39 of 40 random triangles (the 40th was probably missed by the probe rays). Negative coordinates just widen
  the sign+exponent field by one bit.
- Not yet decoded: the index words (0x68..0x77) for 2..5 triangles per block, the per-slot child codes, interior
  nodes, constant-axis cases (desc bit 15 set with a zero width seen when one axis is constant, e.g. a planar cluster),
  and a block stream longer than 92 bytes.

### Fifth pass: several triangles in one block (hand-built blocks decode correctly on the hardware)
- The first triangle of a block is implicitly vertices (0,1,2) = v0 and the first two stream vertices. The stream lists the
  other vertices in first-use order (v0 = vertex 0). Vertex indices are 4 bits, so at most 16 vertices per block.
- Triangle j (j >= 1) has a 14-bit record: 2 low bits, then idx0 | idx1<<4 | idx2<<8 (plain binary, LSB first). The records
  are packed downward from bit 125 of the 16 bytes at 0x68 (bit 0 = byte 0x68 bit 0): record j occupies bits
  126-14j .. 125-14(j-1). Descriptor bits 20..23 = triangles - 1. Setting descriptor bit 15 is needed for 3 or more
  triangles (vendor sets it in most multi-triangle blocks; its exact meaning is open).
- Primitive ids: the block's first triangle returns id = the word at 0x78 minus 0x80000000 (call it p). Triangle j >= 1
  returns (p & ~3) + off_j, with off_j a 2-bit field located just below the lowest record: for N triangles let
  s = 126 - 14(N-1); off_j sits at bits s-2j .. s-2j+1 (b1 highest). Checked for N = 3 and 5. The vendor emits five
  distinct ids in a five-triangle block (e.g. 1,4,0,2,3) which this rule cannot express (offsets are 2 bits), so ids
  are coded some other way for N = 5: open. A block of up to 4 triangles with consecutive ids 4k..4k+3 is expressible.
- Hand-built blocks (re_codec.encode_block) placed in a vendor shell BLAS decode to the right vertices with the right ids.

### Sixth pass: a whole BLAS built from scratch (exec/re_mkblas.py), exact on random soups
`re_mkblas.Builder(tris, header_template).build()` writes a BLAS with no Vulkan structure except the constant first 0x380
header bytes of a one-triangle BLAS. The hardware returns the right closest hit (checked against brute force on 2000
random overlapping triangles and 8000 rays: 0 missed, 0 wrong t, no faults; `exec/re_validate.py`).
- Node (128 bytes): byte 0 = 0x60 (has leaf groups) or 0x40 (only interior slots); bytes 1..3 exponents; 4..15 origin;
  +0x10 w10 = byte offset from the node to its block array (leaf groups); +0x14 = (slot index in the parent) << 28
  (root: 0xf0000000); +0x18 = minus the distance to the PARENT node in 128-byte units (root: 0); +0x1c w1c = distance in
  128-byte units to the node's first interior child; 12 slots from +0x20.
- Parent link: without the right +0x14/+0x18 the hardware loses hits (it resumes the parent's remaining slots through
  the link, a stackless traversal). Zeroed words give garbage, wrong sibling words give faults. The second nibble of
  the top byte of +0x14 (1 in some vendor nodes) is unknown; 0 works.
- Slots: [lo_x hi_x lo_y hi_y lo_z hi_z 0xff code]; empty slot = 00 00 00 00 ff 00 00 00. Code 0x00 = interior child
  (the k-th such slot, counting valid ones, is the node at node + 128*(w1c + k)); leaf groups: one slot per
  triangle of a block with codes (j+1)<<3, the last slot of the group 0x01; the blocks of the groups are consecutive
  from node + w10. In a group the 0x01 slot's box may be tight; the builder uses the union with the later groups.
- Blocks: <= 4 triangles with ids 4k..4k+3 (p = 4k, offsets 1,2,3), descriptor bit 15 set (see fifth pass).
