#!/bin/bash
# Whole Moana island, interactive and fullscreen. Env: GONZALES SCENE CACHE RES
# GONZALES must be a -D WAVEFRONT_BATCH=1 build (the default, n18, is one).
set -euo pipefail

ROOT=$(dirname "$(dirname "$(readlink -f "$0")")")
GONZALES=${GONZALES:-$ROOT/build/gonzales_n18}
SCENE=${SCENE:-/mnt/storage/moana/island/pbrt-v4/island_all.pbrt}
CACHE=${CACHE:-/mnt/storage/gonzales_cache/all.gsc}
RES=${RES:-1920x1080}
export GONZALES_DATA_DIR=${GONZALES_DATA_DIR:-$ROOT/src/gonzales/data}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$ROOT/build

GONZALES_ENV_DOWN=1 GONZALES_PTEX_BUDGET_MB=512 GONZALES_RT_PRUNE=1 \
GONZALES_SCENE_CACHE=$CACHE MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=99 \
  exec "$GONZALES" --interactive --fullscreen --gpu --rt-hardware --resolution "$RES" "$SCENE" "$@"
