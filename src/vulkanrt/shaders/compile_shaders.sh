#!/bin/bash
# Compile GLSL shaders to SPIR-V and generate C headers for embedding.
# Requires: glslc (from shaderc). Ray query needs SPIR-V 1.4+ / Vulkan 1.2,
# hence --target-env=vulkan1.2 (the viewer's rasterization shaders don't
# need this since they only target core 1.0 functionality).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="$SCRIPT_DIR/../generated"
mkdir -p "$OUT_DIR"

compile_shader() {
    local src="$1"
    local name="$2"
    local header="$OUT_DIR/${name}.h"

    echo "Compiling $src -> $header"

    # glslc -mfmt=c outputs a comma-separated list of 32-bit hex words enclosed in braces.
    echo "static const uint32_t ${name}[] = " > "$header"
    glslc --target-env=vulkan1.2 "$src" -mfmt=c -o - >> "$header"
    echo ";" >> "$header"
}

compile_shader "$SCRIPT_DIR/smoke.comp" "smoke_comp_spv"

echo "Done. Generated headers in $OUT_DIR/"
