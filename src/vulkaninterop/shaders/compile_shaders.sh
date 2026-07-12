#!/bin/bash
# Compile GLSL shaders to SPIR-V and generate C headers for embedding.
# Mirrors src/vulkanrt/shaders/compile_shaders.sh's pattern exactly.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="$SCRIPT_DIR/../generated"
mkdir -p "$OUT_DIR"

compile_shader() {
    local src="$1"
    local name="$2"
    local header="$OUT_DIR/${name}.h"

    echo "Compiling $src -> $header"
    echo "static const uint32_t ${name}[] = " > "$header"
    glslc --target-env=vulkan1.2 "$src" -mfmt=c -o - >> "$header"
    echo ";" >> "$header"
}

compile_shader "$SCRIPT_DIR/interop_double.comp" "interop_double_comp_spv"
compile_shader "$SCRIPT_DIR/intersect_batch.comp" "intersect_batch_comp_spv"

echo "Done. Generated headers in $OUT_DIR/"
