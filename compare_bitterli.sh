#!/bin/bash
# Render all Bitterli scenes at 16spp low-res with gonzales + pbrt, then generate HTML.
set -eu

GONZALES=~/work/gonzales/build/gonzales
PBRT=~/src/pbrt-v4/gonsolo/pbrt
BITTERLI=~/src/bitterli
OUT=~/work/gonzales/compare
export LD_LIBRARY_PATH=~/work/gonzales/build${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

SPP=64
MAX_WIDTH=320

mkdir -p "$OUT/images"

SCENES=(
    bathroom bathroom2 bedroom car car2 classroom coffee cornell-box
    curly-hair dining-room dragon furball glass-of-water hair-curl house
    kitchen lamp living-room living-room-2 living-room-3 material-testball
    spaceship staircase staircase2 straight-hair teapot teapot-full
    veach-ajar veach-bidir veach-mis volumetric-caustic water-caustic
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
    scene_file="$BITTERLI/$scene/pbrt/scene-v4.pbrt"

    if [ ! -f "$scene_file" ]; then
        echo "[$idx/$total] SKIP $scene — not found"
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

    # Scoped to the Film directive's own (indented) parameter lines — stops at the
    # next top-level directive — so a LightSource/Texture "string filename" (e.g.
    # an environment map) elsewhere in the file can't be mistaken for the Film's
    # output filename when Film itself has none.
    exr_name=$(awk '/^Film/{infilm=1; print; next} infilm && /^[A-Za-z]/{infilm=0} infilm' "$scene_file" \
        | grep -oP '"string filename"\s*\[?\s*"\K[^"]+\.(exr|png)' | head -1)
    [ -z "$exr_name" ] && exr_name="${scene}.exr"
    # Some scenes' Film filename ends in .png — gonzales/pbrt write it directly
    # (already sRGB); move straight to the -gonzales.png/-pbrt.png output
    # instead of treating it as an .exr that needs tonemapping.
    is_png=false
    [[ "$exr_name" == *.png ]] && is_png=true

    scene_dir="$BITTERLI/$scene/pbrt"
    g_png="$OUT/images/${scene}-gonzales.png"
    p_png="$OUT/images/${scene}-pbrt.png"
    g_time_file="$OUT/images/${scene}-gonzales.time"
    p_time_file="$OUT/images/${scene}-pbrt.time"
    g_mode_file="$OUT/images/${scene}-gonzales.mode"

    ok=true

    # volumetric-caustic's glass-sphere caustic is an SDS path that plain
    # unidirectional NEE path tracing cannot resolve at any sample count —
    # route it through gonzales's --bdpt renderer instead of --gpu. Not
    # applied to every "Integrator bdpt" scene: glass-of-water needs ~33
    # bounces through its nested dielectric surfaces, well past BDPT's
    # hardcoded 10-vertex-per-subpath cap (_BDPT_MAX_VERTS in bdpt.mojo),
    # so it renders black under --bdpt and stays on --gpu, which already
    # handles it correctly.
    use_bdpt=false
    [ "$scene" = "volumetric-caustic" ] && use_bdpt=true

    # ── gonzales ──────────────────────────────────────────────────────────────
    if [ ! -f "$g_png" ]; then
        echo "[$idx/$total] gonzales $scene  ${scale_w}×${scale_h} ${SPP}spp"
        g_exr="$OUT/images/${scene}-gonzales.exr"
        g_log="$OUT/images/${scene}-gonzales.log"
        g_label=""
        if $use_bdpt; then
            if (cd ~/work/gonzales && "$GONZALES" --bdpt --bdpt-spp "$SPP" \
                --resolution "${scale_w}x${scale_h}" "$scene_file") > "$g_log" 2>&1 && \
                [ -f ~/work/gonzales/"$exr_name" ] && \
                { $is_png && mv ~/work/gonzales/"$exr_name" "$g_png" \
                           || { mv ~/work/gonzales/"$exr_name" "$g_exr" && tonemap "$g_exr" "$g_png"; }; }; then
                g_label="BDPT"
            fi
        elif (cd ~/work/gonzales && "$GONZALES" --gpu --spp "$SPP" \
            --resolution "${scale_w}x${scale_h}" "$scene_file") > "$g_log" 2>&1 && \
            [ -f ~/work/gonzales/"$exr_name" ] && \
            { $is_png && mv ~/work/gonzales/"$exr_name" "$g_png" \
                       || { mv ~/work/gonzales/"$exr_name" "$g_exr" && tonemap "$g_exr" "$g_png"; }; }; then
            g_label="GPU"
        fi
        if [ -z "$g_label" ] && ! $use_bdpt; then
            echo "  ! gonzales GPU failed, trying CPU..."
            if (cd ~/work/gonzales && "$GONZALES" --spp "$SPP" \
                --resolution "${scale_w}x${scale_h}" "$scene_file") > "$g_log" 2>&1 && \
                [ -f ~/work/gonzales/"$exr_name" ] && \
                { $is_png && mv ~/work/gonzales/"$exr_name" "$g_png" \
                           || { mv ~/work/gonzales/"$exr_name" "$g_exr" && tonemap "$g_exr" "$g_png"; }; }; then
                g_label="CPU"
            fi
        fi
        if [ -n "$g_label" ]; then
            # Extract rendering time ("Done: X.Xs" from progress line)
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
        p_exr="$OUT/images/${scene}-pbrt.exr"
        p_log="$OUT/images/${scene}-pbrt.log"
        # Patch resolution in a temp copy (pbrt has no --resolution flag)
        lowres_pbrt="$scene_dir/scene-v4-lowres.pbrt"
        sed -e "s/\"integer xresolution\" \[ *[0-9]* *\]/\"integer xresolution\" [ $scale_w ]/g" \
            -e "s/\"integer yresolution\" \[ *[0-9]* *\]/\"integer yresolution\" [ $scale_h ]/g" \
            "$scene_file" > "$lowres_pbrt"
        render_ok=false
        p_label=""
        # pbrt writes PNG directly for scenes whose Film filename ends in .png;
        # in that case move directly to $p_png (already sRGB — do NOT tonemap).
        pbrt_is_png=false
        [[ "$exr_name" == *.png ]] && pbrt_is_png=true
        if (cd "$scene_dir" && "$PBRT" --gpu --spp "$SPP" scene-v4-lowres.pbrt) > "$p_log" 2>&1 \
               && [ -f "$scene_dir/$exr_name" ]; then
            $pbrt_is_png && mv "$scene_dir/$exr_name" "$p_png" \
                         || mv "$scene_dir/$exr_name" "$p_exr"
            render_ok=true
            p_label="GPU"
        else
            echo "  ! pbrt GPU failed, trying CPU..."
            if (cd "$scene_dir" && "$PBRT" --spp "$SPP" scene-v4-lowres.pbrt) > "$p_log" 2>&1 \
                   && [ -f "$scene_dir/$exr_name" ]; then
                $pbrt_is_png && mv "$scene_dir/$exr_name" "$p_png" \
                             || mv "$scene_dir/$exr_name" "$p_exr"
                render_ok=true
                p_label="CPU"
            fi
        fi
        rm -f "$lowres_pbrt"
        if $render_ok; then
            p_t=$(grep -oP '\(\K[\d.]+(?=s[\)|])' "$p_log" | tail -1)
            if $pbrt_is_png || tonemap "$p_exr" "$p_png"; then
                echo "${p_t:-?}" > "$p_time_file"
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
HTML="$OUT/bitterli.html"

cat > "$HTML" << 'HTMLEOF'
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Gonzales vs pbrt — Bitterli scenes</title>
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
<h1>Gonzales vs pbrt — Bitterli scenes</h1>
<p class="subtitle">64 spp · 320px wide · drag slider: left = Gonzales (GPU) · right = pbrt · render times shown below each image</p>
<div class="grid">
HTMLEOF

for scene in "${DONE[@]}"; do
    g_rel="images/${scene}-gonzales.png"
    p_rel="images/${scene}-pbrt.png"
    g_time_file="$OUT/images/${scene}-gonzales.time"
    p_time_file="$OUT/images/${scene}-pbrt.time"
    g_mode_file="$OUT/images/${scene}-gonzales.mode"
    g_t="?"; p_t="?"; g_mode="GPU"
    [ -f "$g_time_file" ] && g_t=$(cat "$g_time_file")
    [ -f "$p_time_file" ] && p_t=$(cat "$p_time_file")
    [ -f "$g_mode_file" ] && g_mode=$(cat "$g_mode_file")
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
      <span class="tp">pbrt: ${p_t}s</span>
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
