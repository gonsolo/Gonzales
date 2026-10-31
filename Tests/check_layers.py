#!/usr/bin/env python3
"""A module may import only from its own layer or the ones below (docs/A5_library_plan.md). Exit status 1 on a new violation."""
import collections, glob, os, re, sys

LAYERS = {
    "core": "geometry rng sampling spectrum rgb2spec noise transform fast_io os_thread progress diagnostics gpu_tuning spd sensors reservoir",
    "scene": "primitives materials curves curve_bvh media nanovdb lights portal_light parse_types lexer ply ply_prefetch mitsuba_serialized ptex_cache "
             "host_textures footprint geom_normal measured_bsdf material_builder light_builder scene_builder scene scene_cache scene_loader pbrt_parser "
             "mitsuba_parser quad_pack bvh rt_build rtcore rt_cache hw_rt xert optixcurves vulkanrt vulkaninterop gpu_bvh",
    "transport": "texturing shade_context manifold render_state medium_interface sss_walk mis_policy bxdf measured_bxdf_eval layered surface sms shadow_media",
    "estimators": "photon_grid",
    "paths": "vcm_strategy vcm_grid vcm_merge vcm_merge_weight vcm_mis bdpt_vertex bdpt_eval bdpt_nee bdpt_connect vcm_camis guide",
    "integrators": "volume_path shading restir_di restir_gi restir_vol restir_sms restir_bdpt sppm bdpt_camera bdpt_light bdpt_render rendering outputs postprocess",
    "gpu": "gpu gpu_scene gpu_media gpu_shade gpu_wavefront gpu_denoise gpu_guide sppm_gpu sppm_kernels vcm_gpu vcm_kernels vcm_rt",
    "app": "pipeline __init__ viewer",
}
# Today's debts. The list only shrinks: docs/A5_library_plan.md, Order.
ALLOWED = set("""
shade_context>guide shade_context>restir_gi



""".split())
IMPORT = re.compile(r"^\s*from\s+(?:\.|gonzales\.)([A-Za-z_0-9]+)\s+import", re.M)


def main():
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "gonzales")
    rank = {m: i for i, ms in enumerate(LAYERS.values()) for m in ms.split()}
    mods = {os.path.basename(f)[:-5]: f for f in glob.glob(os.path.join(src, "*.mojo"))}
    bad = [f"unassigned module: {m}" for m in sorted(set(mods) - set(rank))]
    debts = set()
    for m, f in sorted(mods.items()):
        for x in sorted(set(IMPORT.findall(open(f).read()))):
            if x in mods and m in rank and x in rank and rank[x] > rank[m]:
                edge = f"{m}>{x}"
                debts.add(edge)
                if edge not in ALLOWED:
                    bad.append(f"upward import: {edge}")
    for edge in sorted(ALLOWED - debts):
        bad.append(f"paid debt still listed: {edge}")
    print(f"{len(mods)} modules, {len(debts)} upward imports allowed for now")
    for b in bad:
        print(b)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
