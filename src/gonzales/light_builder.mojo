from std.memory.alloc import unsafe_alloc
from std.math import sqrt
from .lexer import (PbrtScanner, scanner_parse_quoted_string, _psc_collect_params,
                    _psc_streq)
from .parse_types import SceneParseState
from .geometry import RGB
from .transform import transform_points

def _psc_handle_area_light_source(handle: Pointer[PbrtScanner, MutUntrackedOrigin],
                                 s: Pointer[SceneParseState, MutUntrackedOrigin]):
    var sbuf = unsafe_alloc[UInt8](64)
    _ = scanner_parse_quoted_string(handle, sbuf, 64)
    sbuf.unsafe_free()
    var params = _psc_collect_params(handle)
    s[unsafe_offset=0].cur_attr.is_alight = True
    var rgb = params.get_rgb_or_blackbody("L", RGB(Float32(1)))
    var scale = params.get_float("scale", Float32(1.0))
    s[unsafe_offset=0].cur_attr.al_rgb = rgb * scale

def handle_light_source(handle: Pointer[PbrtScanner, MutUntrackedOrigin],
                             s: Pointer[SceneParseState, MutUntrackedOrigin]):
    var ltype = unsafe_alloc[UInt8](64)
    _ = scanner_parse_quoted_string(handle, ltype, 64)
    var params = _psc_collect_params(handle)

    # pbrt convention: "point" lights use "I" (intensity), "distant"/
    # "infinite" use "L" (radiance) -- but check both unconditionally
    # (whichever is present wins, "I" as the final override) rather than
    # gating on light type, matching every light type accepting either name.
    var rgb = params.get_rgb_or_blackbody("L", RGB(Float32(1)))
    rgb = params.get_rgb_or_blackbody("I", rgb)
    var scale = params.get_float("scale", Float32(1.0))
    var xyz = params.get_rgb("from", RGB(Float32(0), Float32(0), Float32(1000)))  # default: from above
    var filename = params.get_string("filename", "")

    if _psc_streq(ltype, "distant"):
        # pbrt defines a distant light by TWO points: "from" (default 0,0,0)
        # and "to" (default 0,0,1); light travels along normalize(to - from),
        # and both points are transformed by the CTM. This used to read only
        # "from", ignore "to" entirely, and default "from" to (0,0,1000) --
        # so any scene that aims its sun the normal way, with "to", got a
        # direction that had nothing to do with what it asked for.
        # disney-cloud gives ONLY "to": pbrt lit it along
        # (-0.583,-0.766,-0.272) while gonzales lit it along (0,0,-1), which
        # is why that cloud came out dim and wrongly shaded.
        var dfrom = params.get_rgb("from", RGB(Float32(0), Float32(0), Float32(0)))
        var dto   = params.get_rgb("to",   RGB(Float32(0), Float32(0), Float32(1)))
        var draw = unsafe_alloc[Float32](8)
        draw[unsafe_offset=0] = dfrom.r; draw[unsafe_offset=1] = dfrom.g; draw[unsafe_offset=2] = dfrom.b; draw[unsafe_offset=3] = Float32(1)
        draw[unsafe_offset=4] = dto.r;   draw[unsafe_offset=5] = dto.g;   draw[unsafe_offset=6] = dto.b;   draw[unsafe_offset=7] = Float32(1)
        var dfin = unsafe_alloc[Float32](8)
        transform_points(s[unsafe_offset=0].ctm.unsafe_ptr(), draw, Int32(2), dfin)
        var ddx = dfin[unsafe_offset=4] - dfin[unsafe_offset=0]
        var ddy = dfin[unsafe_offset=5] - dfin[unsafe_offset=1]
        var ddz = dfin[unsafe_offset=6] - dfin[unsafe_offset=2]
        draw.unsafe_free(); dfin.unsafe_free()
        var dlen = sqrt(ddx*ddx + ddy*ddy + ddz*ddz)
        if dlen < Float32(0.0001):
            ddx = Float32(0); ddy = Float32(0); ddz = Float32(1); dlen = Float32(1)
        # stored as the direction of TRAVEL; _sample_distant_light_nee negates
        # it to get the direction toward the light.
        s[unsafe_offset=0].distant_dirs.append(ddx / dlen)
        s[unsafe_offset=0].distant_dirs.append(ddy / dlen)
        s[unsafe_offset=0].distant_dirs.append(ddz / dlen)
        s[unsafe_offset=0].distant_rgbs.append(rgb.r * scale)
        s[unsafe_offset=0].distant_rgbs.append(rgb.g * scale)
        s[unsafe_offset=0].distant_rgbs.append(rgb.b * scale)
    elif _psc_streq(ltype, "point"):
        # Apply current CTM to position
        var raw = unsafe_alloc[Float32](4)
        raw[unsafe_offset=0] = xyz.r; raw[unsafe_offset=1] = xyz.g; raw[unsafe_offset=2] = xyz.b; raw[unsafe_offset=3] = Float32(1)
        var fin = unsafe_alloc[Float32](4)
        transform_points(s[unsafe_offset=0].ctm.unsafe_ptr(), raw, Int32(1), fin)
        s[unsafe_offset=0].point_pos.append(fin[unsafe_offset=0])
        s[unsafe_offset=0].point_pos.append(fin[unsafe_offset=1])
        s[unsafe_offset=0].point_pos.append(fin[unsafe_offset=2])
        s[unsafe_offset=0].point_rgbs.append(rgb.r * scale)
        s[unsafe_offset=0].point_rgbs.append(rgb.g * scale)
        s[unsafe_offset=0].point_rgbs.append(rgb.b * scale)
        raw.unsafe_free(); fin.unsafe_free()
    elif _psc_streq(ltype, "infinite"):
        if filename != "":
            var file_str = s[unsafe_offset=0].scene_dir + filename
            s[unsafe_offset=0].tex_names.append(String("__inf"))
            s[unsafe_offset=0].tex_files.append(file_str)
            s[unsafe_offset=0].inf_tex_idx.append(Int32(len(s[unsafe_offset=0].tex_names) - 1))
        else:
            s[unsafe_offset=0].inf_tex_idx.append(Int32(-1))
        s[unsafe_offset=0].inf_rgb.append(rgb.r * scale)
        s[unsafe_offset=0].inf_rgb.append(rgb.g * scale)
        s[unsafe_offset=0].inf_rgb.append(rgb.b * scale)
        # Store the light's CTM for env-map direction transform
        for ci in range(16):
            s[unsafe_offset=0].inf_ctm.append(s[unsafe_offset=0].ctm[ci])

    ltype.unsafe_free()
