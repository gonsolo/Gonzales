#!/usr/bin/env python3
"""Generate a synthetic many-light scene designed so ReSTIR DI MUST win
if it is working, and so a tie is evidence of a bug rather than of the
scene being unfavourable.

Design rationale (each choice targets one reason ReSTIR could fail to
show a benefit on the real scenes):

* EQUAL-POWER lights. gonzales selects a light with a power-weighted CDF.
  If powers differ, that CDF is already a decent importance sampler and
  RIS has little to add. Making every light identical in power collapses
  that CDF to uniform selection -- the case ReSTIR exists to beat.

* WIDE SPATIAL SPREAD. With lights scattered over an area far larger than
  the region any one shading point cares about, global (power) ranking
  says nothing about local importance: from a given point, a handful of
  nearby lights dominate and the rest contribute ~nothing. That gap
  between global and local importance is precisely what RIS exploits.

* HEAVY OCCLUSION (optional, --occluders). A slab grid between lights and
  floor so most lights are blocked from most points. This is the direct
  test of the visibility hypothesis: RIS's target function ignores
  visibility, so it will happily resample toward bright-but-blocked
  lights.

* ALL DIFFUSE, DIRECT ONLY. ReSTIR here covers bounce-0 diffuse
  area-light NEE only; maxdepth=1 and diffuse-only surfaces mean 100% of
  the image is inside that scope, so a win cannot be diluted by transport
  the technique never touches.
"""
import argparse, math

def quad(p0, p1, p2, p3):
    """One flat quad as a 2-triangle mesh."""
    pts = " ".join("%.5f %.5f %.5f" % p for p in (p0, p1, p2, p3))
    return ('    Shape "trianglemesh"\n'
            '        "point3 P" [ %s ]\n'
            '        "integer indices" [ 0 1 2 0 2 3 ]\n' % pts)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lights", type=int, default=256, help="light count (a square number is laid out as a grid)")
    ap.add_argument("--occluders", type=int, default=8, help="occluder grid side (0 = none)")
    ap.add_argument("--spp", type=int, default=16)
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--maxdepth", type=int, default=1)
    ap.add_argument("-o", required=True)
    a = ap.parse_args()

    n_side = int(round(math.sqrt(a.lights)))
    n_lights = n_side * n_side
    EXTENT = 40.0        # lights spread over [-EXTENT, EXTENT]^2 ...
    FLOOR = 60.0         # ... above a floor larger still, so edge points see a very skewed light subset
    LIGHT_Y = 12.0
    LIGHT_HALF = 0.35
    OCC_Y = 5.0

    # Equal radiance for every light: with identical emitted power the
    # power-weighted CDF degenerates to uniform selection.
    L = 60.0

    out = []
    out.append('Integrator "path" "integer maxdepth" [ %d ]' % a.maxdepth)
    out.append('Sampler "halton" "integer pixelsamples" [ %d ]' % a.spp)
    out.append('Film "rgb"')
    out.append('    "integer xresolution" [ %d ]' % a.res)
    out.append('    "integer yresolution" [ %d ]' % a.res)
    out.append('    "string filename" [ "manylights.exr" ]')
    # Camera sits BELOW the occluder plane, looking along the floor. Placing
    # it above instead fills the frame with brightly-lit occluder tops and
    # hides the floor -- the receiver is the surface under test, so it has to
    # be what we actually measure.
    out.append('LookAt 0 3 34   0 0 -8   0 1 0')
    out.append('Camera "perspective" "float fov" [ 60 ]')
    out.append('')
    out.append('WorldBegin')
    out.append('')
    out.append('# ── Floor: pure diffuse, so every visible pixel is in ReSTIR\'s scope ──')
    out.append('AttributeBegin')
    out.append('    Material "diffuse" "rgb reflectance" [ 0.6 0.6 0.6 ]')
    out.append(quad((-FLOOR, 0, -FLOOR), (FLOOR, 0, -FLOOR), (FLOOR, 0, FLOOR), (-FLOOR, 0, FLOOR)).rstrip())
    out.append('AttributeEnd')
    out.append('')

    if a.occluders > 0:
        out.append('# ── Occluder grid: blocks most lights from most floor points ──')
        out.append('AttributeBegin')
        out.append('    Material "diffuse" "rgb reflectance" [ 0.25 0.25 0.25 ]')
        m = a.occluders
        step = (2 * EXTENT) / m
        half = step * 0.30            # ~60% areal coverage -> most paths blocked, some open
        for i in range(m):
            for j in range(m):
                cx = -EXTENT + (i + 0.5) * step
                cz = -EXTENT + (j + 0.5) * step
                out.append(quad((cx - half, OCC_Y, cz - half), (cx + half, OCC_Y, cz - half),
                                (cx + half, OCC_Y, cz + half), (cx - half, OCC_Y, cz + half)).rstrip())
        out.append('AttributeEnd')
        out.append('')

    out.append('# ── %d equal-power area lights spread over a wide area ──' % n_lights)
    step = (2 * EXTENT) / n_side
    for i in range(n_side):
        for j in range(n_side):
            cx = -EXTENT + (i + 0.5) * step
            cz = -EXTENT + (j + 0.5) * step
            out.append('AttributeBegin')
            out.append('    AreaLightSource "diffuse" "rgb L" [ %.4f %.4f %.4f ]' % (L, L, L))
            out.append('    Material "diffuse" "rgb reflectance" [ 0 0 0 ]')
            # Winding must make the geometric normal cross(p1-p0, p2-p0) point
            # DOWN at the floor: gonzales accepts a light sample only when
            # cos_l = -dot(light_normal, wi) > 0 with wi pointing surface->light,
            # i.e. a light emits along its normal. The mirrored winding gives
            # +Y and the floor receives nothing at all (verified the hard way).
            out.append(quad((cx - LIGHT_HALF, LIGHT_Y, cz - LIGHT_HALF),
                            (cx + LIGHT_HALF, LIGHT_Y, cz - LIGHT_HALF),
                            (cx + LIGHT_HALF, LIGHT_Y, cz + LIGHT_HALF),
                            (cx - LIGHT_HALF, LIGHT_Y, cz + LIGHT_HALF)).rstrip())
            out.append('AttributeEnd')

    open(a.o, "w").write("\n".join(out) + "\n")
    print("wrote %s: %d lights, %d occluders, maxdepth %d" %
          (a.o, n_lights, a.occluders * a.occluders, a.maxdepth))

main()
