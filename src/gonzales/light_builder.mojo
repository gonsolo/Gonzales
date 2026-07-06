from std.memory import alloc
from std.math import sqrt
from .lexer import (PbrtScanner, scanner_parse_quoted_string,
                    scanner_scan_char, ParamScanner,
                    _psc_streq, _psc_type_is_blackbody,
                    _psc_blackbody_to_rgb, _psc_scan_rgb, _psc_scan_one_float)
from .parse_types import SceneParseState, PSC_FILE_MAX
from .geometry import RGB
from .transform import transform_points

def _psc_handle_area_light_source(handle: UnsafePointer[PbrtScanner, MutAnyOrigin],
                                 s: UnsafePointer[SceneParseState, MutAnyOrigin]):
    var sbuf = alloc[UInt8](64)
    _ = scanner_parse_quoted_string(handle, sbuf, 64)
    s[0].cur_attr.is_alight = True
    s[0].cur_attr.al_rgb = RGB(Float32(1))
    var rgb = alloc[Float32](3)
    rgb[0] = Float32(1); rgb[1] = Float32(1); rgb[2] = Float32(1)
    var scale = Float32(1.0)  # accumulated separately; applied after loop
    var ps = ParamScanner()
    while ps.next(handle):
        if ps.name_is("L") and ps.is_float():
            _psc_scan_rgb(handle, rgb, ps.is_array)
        elif ps.name_is("L") and _psc_type_is_blackbody(ps.type_buf):
            var temp = _psc_scan_one_float(handle, ps.is_array)
            _psc_blackbody_to_rgb(temp, rgb)
        elif ps.name_is("scale") and ps.is_float():
            scale *= _psc_scan_one_float(handle, ps.is_array)
        else:
            ps.skip(handle)
    rgb[0] *= scale; rgb[1] *= scale; rgb[2] *= scale
    s[0].cur_attr.al_rgb = RGB(rgb[0], rgb[1], rgb[2])
    sbuf.free(); rgb.free()

def handle_light_source(handle: UnsafePointer[PbrtScanner, MutAnyOrigin],
                             s: UnsafePointer[SceneParseState, MutAnyOrigin]):
    var ltype = alloc[UInt8](64)
    _ = scanner_parse_quoted_string(handle, ltype, 64)
    var str_val  = alloc[UInt8](PSC_FILE_MAX * 2)
    str_val[0] = UInt8(0)  # must init: alloc doesn't zero-fill
    var rgb      = alloc[Float32](3)
    var xyz      = alloc[Float32](3)
    rgb[0] = Float32(1); rgb[1] = Float32(1); rgb[2] = Float32(1)
    xyz[0] = Float32(0); xyz[1] = Float32(0); xyz[2] = Float32(1000)  # default: from above
    var scale    = Float32(1.0)
    var ps = ParamScanner()
    while ps.next(handle):
        if (ps.name_is("L") or ps.name_is("I")) and ps.is_float():
            _psc_scan_rgb(handle, rgb, ps.is_array)
        elif (ps.name_is("L") or ps.name_is("I")) and _psc_type_is_blackbody(ps.type_buf):
            var temp = _psc_scan_one_float(handle, ps.is_array)
            _psc_blackbody_to_rgb(temp, rgb)
        elif ps.name_is("scale") and ps.is_float():
            scale = _psc_scan_one_float(handle, ps.is_array)
        elif ps.name_is("from") and ps.is_float():
            _psc_scan_rgb(handle, xyz, ps.is_array)   # reuse xyz for point-light position
        elif ps.name_is("filename") and ps.is_str():
            _ = scanner_parse_quoted_string(handle, str_val, PSC_FILE_MAX * 2)
            if ps.is_array:
                _ = scanner_scan_char(handle, UInt8(93))
        else:
            ps.skip(handle)

    if _psc_streq(ltype, "distant"):
        # direction = -from (from describes where light comes from)
        var dlen = sqrt(xyz[0]*xyz[0] + xyz[1]*xyz[1] + xyz[2]*xyz[2])
        if dlen < Float32(0.0001): dlen = Float32(1.0)
        s[0].distant_dirs.append(-xyz[0] / dlen)
        s[0].distant_dirs.append(-xyz[1] / dlen)
        s[0].distant_dirs.append(-xyz[2] / dlen)
        s[0].distant_rgbs.append(rgb[0] * scale)
        s[0].distant_rgbs.append(rgb[1] * scale)
        s[0].distant_rgbs.append(rgb[2] * scale)
    elif _psc_streq(ltype, "point"):
        # Apply current CTM to position
        var raw = alloc[Float32](4)
        raw[0] = xyz[0]; raw[1] = xyz[1]; raw[2] = xyz[2]; raw[3] = Float32(1)
        var fin = alloc[Float32](4)
        transform_points(s[0].ctm.unsafe_ptr(), raw, Int32(1), fin)
        s[0].point_pos.append(fin[0])
        s[0].point_pos.append(fin[1])
        s[0].point_pos.append(fin[2])
        s[0].point_rgbs.append(rgb[0] * scale)
        s[0].point_rgbs.append(rgb[1] * scale)
        s[0].point_rgbs.append(rgb[2] * scale)
        raw.free(); fin.free()
    elif _psc_streq(ltype, "infinite"):
        if str_val[0] != UInt8(0):
            var file_str = s[0].scene_dir + String(unsafe_from_utf8_ptr=str_val.as_immutable())
            s[0].tex_names.append(String("__inf"))
            s[0].tex_files.append(file_str)
            s[0].inf_tex_idx.append(Int32(len(s[0].tex_names) - 1))
        else:
            s[0].inf_tex_idx.append(Int32(-1))
        s[0].inf_rgb.append(rgb[0] * scale)
        s[0].inf_rgb.append(rgb[1] * scale)
        s[0].inf_rgb.append(rgb[2] * scale)
        # Store the light's CTM for env-map direction transform
        for ci in range(16):
            s[0].inf_ctm.append(s[0].ctm[ci])

    ltype.free(); str_val.free(); rgb.free(); xyz.free()
