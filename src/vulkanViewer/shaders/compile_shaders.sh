#!/bin/bash
# Compile GLSL shaders to SPIR-V and generate C headers for embedding.
# Requires: glslc (from shaderc).
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
    glslc "$src" -mfmt=c -o - >> "$header"
    echo ";" >> "$header"
}

compile_shader "$SCRIPT_DIR/fullscreen.vert" "fullscreen_vert_spv"
compile_shader "$SCRIPT_DIR/fullscreen.frag" "fullscreen_frag_spv"

echo "Done. Generated headers in $OUT_DIR/"
