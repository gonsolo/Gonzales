# Register budgets for the hottest wavefront kernels. nvvm.minctasm = N asks ptxas for N resident blocks per SM at the
# block size the kernel is launched with (256), i.e. at most 65536 / (N * 256) registers per thread: 1 = no cap (255),
# 2 = 128, 3 = 85. A codegen hint only: output is bit-identical, the cost is local-memory spills when a kernel needs
# more. Measured on pavilion, kitchen and staircase (PT, 64 spp): 2 is 25-33% faster than 1, 3 is 15-20% faster than 1,
# mixes of 2 and 3 are no better than 2 everywhere (docs/A4_execution_modes_plan.md, step 5).
comptime MINCTA_NEE = 2
comptime MINCTA_DIFFUSE = 2
comptime MINCTA_COATED = 2
comptime MINCTA_CONDUCTOR = 2
comptime MINCTA_MEASURED = 2
comptime MINCTA_DIELECTRIC = 2
comptime MINCTA_TRAVERSE = 2
comptime MINCTA_VCM_EMIT = 2
comptime MINCTA_VCM_SPLAT = 2
comptime MINCTA_VCM_CONNECT = 2
comptime MINCTA_VCM_LBOUNCE = 2
comptime MINCTA_VCM_CBOUNCE = 2
comptime MINCTA_VCM_RESOLVE = 2
