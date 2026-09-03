from std.memory import alloc
from std.math import sqrt
from .lexer import (PbrtScanner, scanner_parse_quoted_string, _psc_collect_params,
                    _psc_streq)
from .parse_types import SceneParseState
from .geometry import RGB
from .transform import transform_points

def _psc_handle_area_light_source(handle: UnsafePointer[PbrtScanner, MutExternalOrigin],
                                 s: UnsafePointer[SceneParseState, MutExternalOrigin]):
    var sbuf = alloc[UInt8](64)
    _ = scanner_parse_quoted_string(handle, sbuf, 64)
    sbuf.free()
    var params = _psc_collect_params(handle)
    s[0].cur_attr.is_alight = True
    var rgb = params.get_rgb_or_blackbody("L", RGB(Float32(1)))
    var scale = params.get_float("scale", Float32(1.0))
    s[0].cur_attr.al_rgb = rgb * scale

def handle_light_source(handle: UnsafePointer[PbrtScanner, MutExternalOrigin],
                             s: UnsafePointer[SceneParseState, MutExternalOrigin]):
    var ltype = alloc[UInt8](64)
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
        # direction = -from (from describes where light comes from)
        var dlen = sqrt(xyz.r*xyz.r + xyz.g*xyz.g + xyz.b*xyz.b)
        if dlen < Float32(0.0001): dlen = Float32(1.0)
        s[0].distant_dirs.append(-xyz.r / dlen)
        s[0].distant_dirs.append(-xyz.g / dlen)
        s[0].distant_dirs.append(-xyz.b / dlen)
        s[0].distant_rgbs.append(rgb.r * scale)
        s[0].distant_rgbs.append(rgb.g * scale)
        s[0].distant_rgbs.append(rgb.b * scale)
    elif _psc_streq(ltype, "point"):
        # Apply current CTM to position
        var raw = alloc[Float32](4)
        raw[0] = xyz.r; raw[1] = xyz.g; raw[2] = xyz.b; raw[3] = Float32(1)
        var fin = alloc[Float32](4)
        transform_points(s[0].ctm.unsafe_ptr(), raw, Int32(1), fin)
        s[0].point_pos.append(fin[0])
        s[0].point_pos.append(fin[1])
        s[0].point_pos.append(fin[2])
        s[0].point_rgbs.append(rgb.r * scale)
        s[0].point_rgbs.append(rgb.g * scale)
        s[0].point_rgbs.append(rgb.b * scale)
        raw.free(); fin.free()
    elif _psc_streq(ltype, "infinite"):
        if filename != "":
            var file_str = s[0].scene_dir + filename
            s[0].tex_names.append(String("__inf"))
            s[0].tex_files.append(file_str)
            s[0].inf_tex_idx.append(Int32(len(s[0].tex_names) - 1))
        else:
            s[0].inf_tex_idx.append(Int32(-1))
        s[0].inf_rgb.append(rgb.r * scale)
        s[0].inf_rgb.append(rgb.g * scale)
        s[0].inf_rgb.append(rgb.b * scale)
        # Store the light's CTM for env-map direction transform
        for ci in range(16):
            s[0].inf_ctm.append(s[0].ctm[ci])

    ltype.free()
