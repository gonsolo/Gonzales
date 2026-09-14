#!/bin/bash
# Wall/CPU/render/startup timing over a fixed scene set, for regression tracking.
# Usage: ./bench.sh [scene ...]            (default: every scene in SCENE_FILES)
# Env overrides: GONZALES PBRT_DIR SPP RES FLAGS RESULTS REV
set -euo pipefail

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
GONZALES=${GONZALES:-$SCRIPT_DIR/build/gonzales}
PBRT_DIR=${PBRT_DIR:-$SCRIPT_DIR/Scenes/pbrt-v4-scenes}
SPP=${SPP:-16}
RES=${RES:-512x512}
FLAGS=${FLAGS:---gpu}
RESULTS=${RESULTS:-$SCRIPT_DIR/build/bench-results.tsv}
REV=${REV:-$(git -C "$SCRIPT_DIR" describe --always --dirty 2>/dev/null || echo unknown)}
# gonzales resolves its data dir relative to the cwd by default, and we run it
# from a scratch dir below.
export GONZALES_DATA_DIR=${GONZALES_DATA_DIR:-$SCRIPT_DIR/src/gonzales/data}
# Appended, not prepended, so a caller's LD_LIBRARY_PATH can shadow build/ libs.
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$SCRIPT_DIR/build

declare -A SCENE_FILES=(
  [sanmiguel]=sanmiguel/sanmiguel-courtyard.pbrt
  [bistro]=bistro/bistro_cafe.pbrt
  [ganesha]=ganesha/ganesha.pbrt
  [bunny-fur]=bunny-fur/bunny-fur.pbrt
  [disney-cloud]=disney-cloud/disney-cloud.pbrt
)
if [ $# -gt 0 ]; then
    SCENES=("$@")
else
    SCENES=(sanmiguel bistro ganesha bunny-fur disney-cloud)
fi
read -ra flag_args <<< "$FLAGS"

stamp=$(date -Iseconds)
mkdir -p "$(dirname "$RESULTS")"
[ -f "$RESULTS" ] || printf 'date\trev\tflags\tspp\tres\tscene\twall_s\tcpu_s\trender_s\tstartup_s\n' > "$RESULTS"

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
TIMEFORMAT='%R %U %S'

printf '%-14s %8s %8s %9s %10s\n' scene wall_s cpu_s render_s startup_s
for scene in "${SCENES[@]}"; do
    file="$PBRT_DIR/${SCENE_FILES[$scene]:-}"
    if [ ! -f "$file" ]; then
        echo "$scene: not found ($file)" >&2
        continue
    fi
    log="$work/$scene.log"
    # cd into a scratch dir so the output image doesn't land in the repo.
    if ! { time (cd "$work" && "$GONZALES" "${flag_args[@]}" --no-denoise \
            --spp "$SPP" --resolution "$RES" "$file" > "$log" 2>&1); } 2> "$work/time"; then
        echo "$scene: FAILED" >&2
        tail -5 "$log" >&2
        continue
    fi
    read -r wall user sys < "$work/time"
    render=$(tr '\r' '\n' < "$log" | grep -oP 'Done: \K[\d.]+' | tail -1 || true)
    # gonzales can bail out early with exit status 0 (e.g. missing data files),
    # so a run that never reported a render time counts as a failure.
    if [ -z "$render" ]; then
        echo "$scene: FAILED (no render time reported)" >&2
        tail -5 "$log" >&2
        continue
    fi
    cpu=$(awk -v u="$user" -v s="$sys" 'BEGIN { printf "%.2f", u + s }')
    startup=$(awk -v w="$wall" -v r="$render" 'BEGIN { printf "%.2f", w - r }')
    printf '%-14s %8.2f %8s %9s %10s\n' "$scene" "$wall" "$cpu" "$render" "$startup"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$stamp" "$REV" "$FLAGS" "$SPP" "$RES" \
        "$scene" "$wall" "$cpu" "$render" "$startup" >> "$RESULTS"
done
