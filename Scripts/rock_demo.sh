#!/bin/bash
# Moana's lava rock in a window on the Intel B580: tessellation-free displaced Catmull-Clark patches (Tools/disp_proto.mojo).
# Fly with W/A/S/D, Q/E (Space/C), Shift x10, wheel = speed, drag = look; +/- displacement x2 / x0.5, 0 displacement off/on,
# L headlight, Esc quit. Env: EXAG (displacement scale, default 1) W H EXPOSURE COLOR (colour Ptex) HALF=0 (full res moving)
# GRID=4/8/16 (sub-squares a patch side in the BVH, default 16) LEAF=exact/box/texel (exact bilinear leaves at a texel or pixel; footprint boxes; texel only)
# DEMO_FRAMES=N (no window: N frames around the rock, then 32 still ones to rock/demo_last.ppm). Build the tool first:
#   make unittest-intel-src && . $HAL/env.sh && uv run mojo build Tools/disp_proto.mojo -I build/intel-src \
#     -o build/disp_proto_gpu --target-accelerator intel:xe2 <MOJO_LINK_FLAGS>
set -euo pipefail

ROOT=$(dirname "$(dirname "$(readlink -f "$0")")")
TOOL=${TOOL:-$ROOT/build/disp_proto_gpu}
HAL=${HAL:-/mnt/storage/gonzales_cache/mojo-src-hal}
set +u; . "$HAL/env.sh"; set -u
export MODULAR_DRIVER_PLUGINS="level_zero@$HAL/lib/plugins/LevelZeroPlugin.so;cpu@$HAL/lib/plugins/CPUPlugin.so"
export LD_LIBRARY_PATH=$ROOT/build:/mnt/storage/neo-ws/build/bin:/mnt/storage/rt-ws/build:$HOME/src/igc-ws/build/IGC/Release${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export NEOReadDebugKeys=1 ForcePrivateMemoryPerDispatch=1 NEO_CACHE_DIR=${NEO_CACHE_DIR:-/mnt/work/acct/neo}
[ -n "${DEMO_FRAMES:-}" ] || export INTERACTIVE=1
exec "$TOOL"
