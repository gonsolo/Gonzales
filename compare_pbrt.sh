#!/bin/bash
# Render pbrt-v4-scenes at low-res with gonzales + pbrt, generate compare/pbrt.html.
set -eu

GONZALES=~/work/gonzales/build/gonzales
PBRT=~/src/pbrt-v4/gonsolo/pbrt
PBRT_DIR=~/work/gonzales/Scenes/pbrt-v4-scenes
OUT=~/work/gonzales/compare
export LD_LIBRARY_PATH=~/work/gonzales/build${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

SPP=64
MAX_WIDTH=320

mkdir -p "$OUT/images"

# scene_name → entry pbrt file (relative to PBRT_DIR/$scene/)
declare -A SCENE_FILES
SCENE_FILES=(
  [barcelona-pavilion]="pavilion-day.pbrt"
  [bistro]="bistro_cafe.pbrt"
  [bmw-m6]="bmw-m6.pbrt"
  [bunny-cloud]="bunny-cloud.pbrt"
  [bunny-fur]="bunny-fur.pbrt"
  [contemporary-bathroom]="contemporary-bathroom.pbrt"
  [crown]="crown.pbrt"
  [dambreak]="dambreak0.pbrt"
  [disney-cloud]="disney-cloud.pbrt"
  [explosion]="explosion.pbrt"
  [ganesha]="ganesha.pbrt"
  [hair]="hair-actual-bsdf.pbrt"
  [head]="head.pbrt"
  [killeroos]="killeroo-coated-gold.pbrt"
  [kroken]="camera-1.pbrt"
  [landscape]="view-0.pbrt"
  [lte-orb]="lte-orb-rough-glass.pbrt"
  [pbrt-book]="book.pbrt"
  [sanmiguel]="sanmiguel-courtyard.pbrt"
  [smoke-plume]="plume.pbrt"
  [sportscar]="sportscar-sky.pbrt"
  [sssdragon]="dragon_10.pbrt"
  [transparent-machines]="frame1266.pbrt"
  [villa]="villa-daylight.pbrt"
  [watercolor]="camera-10.pbrt"
  [zero-day]="frame120.pbrt"
)

SCENES=(
  barcelona-pavilion bistro bmw-m6 bunny-cloud bunny-fur
  contemporary-bathroom crown dambreak disney-cloud explosion
  ganesha hair head killeroos kroken landscape lte-orb
  pbrt-book sanmiguel smoke-plume sportscar sssdragon
  transparent-machines villa watercolor zero-day
)

tonemap() {
    local src="$1" dst="$2"
    oiiotool "$src" --ociodisplay "sRGB - Display" "ACES 2.0 - SDR 100 nits (Rec.709)" -o "$dst"
}

DONE=()
FAILED=()
total=${#SCENES[@]}
idx=0

for scene in "${SCENES[@]}"; do
    idx=$((idx + 1))
    scene_pbrt="${SCENE_FILES[$scene]:-}"
    scene_dir="$PBRT_DIR/$scene"
    scene_file="$scene_dir/$scene_pbrt"

    if [ ! -f "$scene_file" ]; then
        echo "[$idx/$total] SKIP $scene — $scene_file not found"
        FAILED+=("$scene")
        continue
    fi

    # Scale to MAX_WIDTH, preserve aspect ratio
    xres=$(grep -oP '"integer xresolution"\s*\[\s*\K\d+' "$scene_file" | head -1)
    yres=$(grep -oP '"integer yresolution"\s*\[\s*\K\d+' "$scene_file" | head -1)
    xres=${xres:-512}; yres=${yres:-512}
    scale_w=$MAX_WIDTH
    scale_h=$(( yres * scale_w / xres ))
    [ "$scale_h" -lt 1 ] && scale_h=1

    # Match Film "string filename" (bracket or plain form) from the Film block only.
    # Scoped to the Film directive's own (indented) parameter lines — stops at the
    # next top-level directive — so a LightSource/Texture "string filename" (e.g.
    # an environment map like textures/sky.exr) elsewhere in the file can't be
    # mistaken for the Film's output filename when Film itself has none.
    exr_name=$(awk '/^Film/{infilm=1; print; next} infilm && /^[A-Za-z]/{infilm=0} infilm' "$scene_file" \
        | grep -oP '"string filename"\s*\[?\s*"\K[^"]+\.exr' | head -1)
    g_exr_name="${exr_name:-gonzales.exr}"   # gonzales default when Film has no filename
    p_exr_name="${exr_name:-pbrt.exr}"       # pbrt default when Film has no filename

    g_png="$OUT/images/pbrt-${scene}-gonzales.png"
    p_png="$OUT/images/pbrt-${scene}-pbrt.png"
    g_time_file="$OUT/images/pbrt-${scene}-gonzales.time"
    p_time_file="$OUT/images/pbrt-${scene}-pbrt.time"
    g_mode_file="$OUT/images/pbrt-${scene}-gonzales.mode"
    p_mode_file="$OUT/images/pbrt-${scene}-pbrt.mode"

    ok=true

    # barcelona-pavilion's trees are placed via ObjectInstance — gonzales only
    # supports object instancing (two-level BVH: BLAS per template + TLAS
    # instance leaves) on the CPU render path so far (--gpu's device-side
    # scene upload doesn't carry instance data yet, a documented follow-up
    # gap). Force CPU-only here so the fix actually shows up in the compare;
    # --gpu would otherwise "succeed" while silently still missing the trees.
    force_cpu=false
    case "$scene" in
        barcelona-pavilion) force_cpu=true ;;
    esac

    # ── gonzales ──────────────────────────────────────────────────────────────
    if [ ! -f "$g_png" ]; then
        echo "[$idx/$total] gonzales $scene  ${scale_w}×${scale_h} ${SPP}spp"
        g_exr="$OUT/images/pbrt-${scene}-gonzales.exr"
        g_log="$OUT/images/pbrt-${scene}-gonzales.log"
        g_label=""
        if [ "$force_cpu" = true ]; then
            if (cd ~/work/gonzales && "$GONZALES" --spp "$SPP" \
                --resolution "${scale_w}x${scale_h}" "$scene_file") > "$g_log" 2>&1 && \
                [ -f ~/work/gonzales/"$g_exr_name" ] && \
                mv ~/work/gonzales/"$g_exr_name" "$g_exr" && \
                tonemap "$g_exr" "$g_png"; then
                g_label="CPU"
            fi
        elif (cd ~/work/gonzales && "$GONZALES" --gpu --spp "$SPP" \
            --resolution "${scale_w}x${scale_h}" "$scene_file") > "$g_log" 2>&1 && \
            [ -f ~/work/gonzales/"$g_exr_name" ] && \
            mv ~/work/gonzales/"$g_exr_name" "$g_exr" && \
            tonemap "$g_exr" "$g_png"; then
            g_label="GPU"
        else
            echo "  ! gonzales GPU failed, trying CPU..."
            if (cd ~/work/gonzales && "$GONZALES" --spp "$SPP" \
                --resolution "${scale_w}x${scale_h}" "$scene_file") > "$g_log" 2>&1 && \
                [ -f ~/work/gonzales/"$g_exr_name" ] && \
                mv ~/work/gonzales/"$g_exr_name" "$g_exr" && \
                tonemap "$g_exr" "$g_png"; then
                g_label="CPU"
            fi
        fi
        if [ -n "$g_label" ]; then
            g_t=$(grep -oP "Done: \K[\d.]+" "$g_log" | tail -1)
            echo "${g_t:-?}" > "$g_time_file"
            echo "$g_label" > "$g_mode_file"
            echo "  → gonzales OK ($g_label, ${g_t:-?}s render)"
        else
            echo "  ✗ gonzales FAILED (see $g_log)"; ok=false
        fi
    else
        echo "[$idx/$total] gonzales $scene (cached)"
    fi

    # ── pbrt ──────────────────────────────────────────────────────────────────
    if [ ! -f "$p_png" ]; then
        echo "[$idx/$total] pbrt     $scene  ${scale_w}×${scale_h} ${SPP}spp"
        p_exr="$OUT/images/pbrt-${scene}-pbrt.exr"
        p_log="$OUT/images/pbrt-${scene}-pbrt.log"
        lowres_pbrt="$scene_dir/${scene_pbrt%.pbrt}-lowres.pbrt"
        sed -e "s/\"integer xresolution\" \[ *[0-9]* *\]/\"integer xresolution\" [ $scale_w ]/g" \
            -e "s/\"integer yresolution\" \[ *[0-9]* *\]/\"integer yresolution\" [ $scale_h ]/g" \
            "$scene_file" > "$lowres_pbrt"
        render_ok=false
        p_label=""
        lowres_base=$(basename "$lowres_pbrt")
        if (cd "$scene_dir" && "$PBRT" --gpu --spp "$SPP" "$lowres_base") > "$p_log" 2>&1 \
               && [ -f "$scene_dir/$p_exr_name" ]; then
            mv "$scene_dir/$p_exr_name" "$p_exr"
            render_ok=true
            p_label="GPU"
        else
            echo "  ! pbrt GPU failed, trying CPU..."
            if (cd "$scene_dir" && "$PBRT" --spp "$SPP" "$lowres_base") > "$p_log" 2>&1 \
                   && [ -f "$scene_dir/$p_exr_name" ]; then
                mv "$scene_dir/$p_exr_name" "$p_exr"
                render_ok=true
                p_label="CPU"
            fi
        fi
        rm -f "$lowres_pbrt"
        if $render_ok; then
            if tonemap "$p_exr" "$p_png"; then
                p_t=$(grep -oP '\(\K[\d.]+(?=s[\)|])' "$p_log" | tail -1)
                echo "${p_t:-?}" > "$p_time_file"
                echo "$p_label" > "$p_mode_file"
                echo "  → pbrt OK ($p_label, ${p_t:-?}s render)"
            else
                echo "  ✗ tonemap failed"; ok=false
            fi
        else
            echo "  ✗ pbrt FAILED (see $p_log)"
            ok=false
        fi
    else
        echo "[$idx/$total] pbrt     $scene (cached)"
    fi

    if $ok && [ -f "$g_png" ] && [ -f "$p_png" ]; then
        DONE+=("$scene")
    else
        FAILED+=("$scene")
    fi
done

echo ""
echo "Done: ${#DONE[@]}  Failed: ${#FAILED[@]}"
[ ${#FAILED[@]} -gt 0 ] && echo "Failed: ${FAILED[*]}"

# ── Generate HTML ─────────────────────────────────────────────────────────────
HTML="$OUT/pbrt.html"

cat > "$HTML" << 'HTMLEOF'
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Gonzales vs pbrt — pbrt-v4 scenes</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #111; color: #ddd; font-family: system-ui, sans-serif; padding: 24px; }
h1 { font-size: 1.4rem; font-weight: 600; margin-bottom: 4px; }
.subtitle { color: #888; font-size: 0.85rem; margin-bottom: 28px; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(500px, 1fr)); gap: 28px; }
.card { background: #1a1a1a; border-radius: 8px; overflow: hidden; }
.card-title { font-size: 0.9rem; font-weight: 500; padding: 10px 14px; color: #bbb; letter-spacing: 0.04em; }
.timebar { display: flex; gap: 16px; padding: 6px 14px 10px; font-size: 0.78rem; color: #888; }
.timebar .tg { color: #7cf; }
.timebar .tp { color: #fc7; }

/* ── comparison slider ── */
.cmp { position: relative; overflow: hidden; cursor: col-resize; user-select: none; display: block; }
.cmp img { display: block; width: 100%; height: auto; pointer-events: none; }
.cmp .overlay { position: absolute; inset: 0; overflow: hidden; }
.cmp .overlay img { position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: cover; }
.cmp .divider { position: absolute; top: 0; bottom: 0; width: 2px; background: #fff; transform: translateX(-50%); pointer-events: none; }
.cmp .handle {
  position: absolute; top: 50%; transform: translate(-50%, -50%);
  width: 40px; height: 40px; border-radius: 50%;
  background: #fff; display: flex; align-items: center; justify-content: center;
  box-shadow: 0 1px 6px rgba(0,0,0,.6); pointer-events: none;
}
.cmp .handle::before, .cmp .handle::after {
  content: ''; display: block; width: 0; height: 0;
  border-top: 6px solid transparent; border-bottom: 6px solid transparent;
}
.cmp .handle::before { border-right: 8px solid #333; margin-right: 4px; }
.cmp .handle::after  { border-left:  8px solid #333; }
.lbl { position: absolute; top: 8px; font-size: 0.72rem; font-weight: 700;
       padding: 2px 7px; border-radius: 3px; letter-spacing: 0.05em; pointer-events: none; }
.lbl-l { left: 8px;  background: rgba(0,0,0,.6); color: #7cf; }
.lbl-r { right: 8px; background: rgba(0,0,0,.6); color: #fc7; }
</style>
</head>
<body>
<h1>Gonzales vs pbrt — pbrt-v4 scenes</h1>
<p class="subtitle">64 spp · 320px wide · drag slider: left = Gonzales (GPU) · right = pbrt · render times shown below each image</p>
<div class="grid">
HTMLEOF

for scene in "${DONE[@]}"; do
    g_rel="images/pbrt-${scene}-gonzales.png"
    p_rel="images/pbrt-${scene}-pbrt.png"
    g_time_file="$OUT/images/pbrt-${scene}-gonzales.time"
    p_time_file="$OUT/images/pbrt-${scene}-pbrt.time"
    g_mode_file="$OUT/images/pbrt-${scene}-gonzales.mode"
    p_mode_file="$OUT/images/pbrt-${scene}-pbrt.mode"
    g_t="?"; p_t="?"; g_mode="GPU"; p_mode="GPU"
    [ -f "$g_time_file" ] && g_t=$(cat "$g_time_file")
    [ -f "$p_time_file" ] && p_t=$(cat "$p_time_file")
    [ -f "$g_mode_file" ] && g_mode=$(cat "$g_mode_file")
    [ -f "$p_mode_file" ] && p_mode=$(cat "$p_mode_file")
    cat >> "$HTML" << SCENE
  <div class="card">
    <div class="card-title">${scene}</div>
    <div class="cmp">
      <img src="${g_rel}" alt="Gonzales">
      <div class="overlay"><img src="${p_rel}" alt="pbrt"></div>
      <div class="divider"></div>
      <div class="handle"></div>
      <span class="lbl lbl-l">GONZALES</span>
      <span class="lbl lbl-r">PBRT</span>
    </div>
    <div class="timebar">
      <span class="tg">Gonzales ${g_mode}: ${g_t}s</span>
      <span class="tp">pbrt ${p_mode}: ${p_t}s</span>
    </div>
  </div>
SCENE
done

cat >> "$HTML" << 'HTMLEOF'
</div>
<script>
document.querySelectorAll('.cmp').forEach(el => {
  const overlay = el.querySelector('.overlay');
  const divider = el.querySelector('.divider');
  const handle  = el.querySelector('.handle');
  let dragging = false;

  function set(p) {
    p = Math.max(0, Math.min(100, p));
    overlay.style.clipPath = `inset(0 0 0 ${p}%)`;
    divider.style.left = p + '%';
    handle.style.left  = p + '%';
  }
  set(50);

  function pos(e) {
    const r = el.getBoundingClientRect();
    const x = (e.touches ? e.touches[0].clientX : e.clientX) - r.left;
    return x / r.width * 100;
  }

  el.addEventListener('mousedown',  e => { dragging = true; set(pos(e)); e.preventDefault(); });
  el.addEventListener('touchstart', e => { dragging = true; set(pos(e)); }, { passive: true });
  window.addEventListener('mousemove',  e => { if (dragging) set(pos(e)); });
  window.addEventListener('touchmove',  e => { if (dragging) set(pos(e)); }, { passive: true });
  window.addEventListener('mouseup',  () => dragging = false);
  window.addEventListener('touchend', () => dragging = false);
});
</script>
</body>
</html>
HTMLEOF

echo ""
echo "HTML: $HTML"
echo "Open: xdg-open $HTML"
