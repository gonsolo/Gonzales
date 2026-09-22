"""Generate the ROUGH-COAT energy compensation table.

    make coat_energy_table && build/coat_energy_table > /tmp/coat_e.csv

`lobe_eval`'s coated_walk branch evaluates a coateddiffuse vertex with
`coat_eval_smooth`, the SMOOTH-coat closed form, whatever the coat's
roughness. The stochastic coat walk that the SAMPLED strategies use loses
energy a smooth coat does not -- rough facets that refract back below the
surface and recycle, Beer-Lambert along tilted facet paths, Smith masking on
both crossings. So every EVALUATED strategy (VCM's connections, merging, NEE)
carries more energy than the walk actually transports, and on a multi-bounce
scene that per-bounce excess compounds.

This prints, per (mu_o, alpha):

    E_walk  the directional albedo the WALK delivers through the base
            (COAT_EXIT paths only -- the coat's own specular reflection is a
            separate lobe that does not go through the evaluator)
    E_eval  the same quantity for the EVALUATOR, integral of
            coat_eval_smooth(wo,wi) cos_i dwi
    ratio   E_walk / E_eval, i.e. what the evaluator must be multiplied by

with a WHITE base, which is what makes E_eval clean: at alb = 1 the
`gain_cw` normalisation inside lobe_eval cancels exactly (its numerator and
denominator become the same `1 - fdr_moment`), so the evaluated lobe reduces
to coat_eval_smooth itself and the two columns are directly comparable.

E_eval is a deterministic QUADRATURE, never a sampler. That is not fussiness:
ggx_albedo's docstring records that the first GGX table was built as an
average over VNDF samples and silently inherited a bug in sample_ggx_vndf,
giving a table that was right only at mu = 1. A quadrature cannot inherit a
sampler's bug because it never calls one. E_walk has to be Monte Carlo --
the walk IS a stochastic process and is the thing being matched -- so it is
the half that needs validating downstream, against an independent
PT-vs-connections probe rather than against this table's own numbers.

ratio -> 1 as alpha -> 0 is the built-in self-check: at alpha 0 the walk and
the smooth closed form model the same material.

A fifth column, `corr = E_walk(mu,alpha) / E_walk(mu,0)`, is the actual
compensation factor to bake into lobe_eval's coated_walk branch: normalising
by the alpha=0 column (not by E_eval) is what keeps a SMOOTH coat bit-
unchanged and only corrects the ROUGH excess. mu and alpha are both a
UNIFORM 0.0..1.0 grid, 11 points each, so the GPU-side lookup can index with
plain floor(mu*10)/floor(alpha*10) instead of searching unevenly spaced
knots.
"""
from std.math import sqrt, sin, cos, min, max
from gonzales.geometry import Vec3f, RGB, dot, PI
from gonzales.rng import PCG32
from gonzales.bxdf import (
    CoatWalk, coat_walk_begin, coat_walk_enter, coat_walk_at_base,
    coat_walk_scatter, coat_eval_smooth,
    COAT_WALKING, COAT_REFLECT, COAT_EXIT, COAT_ABSORB,
)

comptime IOR = Float32(1.5)
comptime N_WALK = 400000        # Monte Carlo trials per grid cell
comptime N_THETA = 256          # quadrature resolution, polar
comptime N_PHI = 64             # quadrature resolution, azimuth


def walk_albedo(mu_o: Float32, alpha: Float32, seed: UInt64) -> Float32:
    """Directional albedo of the BASE-transport lobe, as the walk delivers it.

    Only COAT_EXIT contributes. COAT_REFLECT is the coat's own specular
    bounce: a different lobe, handled elsewhere, and not what the evaluator
    being corrected here models. The entry Fresnel needs no 1/(1-F): the
    coin flip in coat_walk_enter IS its estimator, so averaging beta over ALL
    trials (zero for the ones that reflect or are absorbed) already carries
    it."""
    var gn = Vec3f(Float32(0), Float32(0), Float32(1))
    var st = sqrt(max(Float32(0), Float32(1) - mu_o * mu_o))
    var wo = Vec3f(st, Float32(0), mu_o)
    var alb = RGB(Float32(1))
    var pcg = PCG32(seed, 1)
    var acc = Float32(0)
    for _ in range(N_WALK):
        var cw = coat_walk_begin(gn, wo, alb, IOR, alpha, pcg)
        coat_walk_enter(cw, pcg)
        if cw.event == COAT_ABSORB or cw.event == COAT_REFLECT:
            continue
        while cw.event == COAT_WALKING:
            if not coat_walk_at_base(cw, pcg):
                break
            coat_walk_scatter(cw, pcg)
        if cw.event == COAT_EXIT:
            acc += (cw.beta.r + cw.beta.g + cw.beta.b) * Float32(1.0 / 3.0)
    return acc / Float32(N_WALK)


def eval_albedo(mu_o: Float32) -> Float32:
    """Integral of coat_eval_smooth(wo,wi) cos_i over the hemisphere, by
    deterministic quadrature. Independent of alpha -- which is precisely the
    defect being measured."""
    var gn = Vec3f(Float32(0), Float32(0), Float32(1))
    var st = sqrt(max(Float32(0), Float32(1) - mu_o * mu_o))
    var wo = Vec3f(st, Float32(0), mu_o)
    var acc = Float32(0)
    var dth = (Float32(0.5) * PI) / Float32(N_THETA)
    var dph = (Float32(2.0) * PI) / Float32(N_PHI)
    for it in range(N_THETA):
        var th = (Float32(it) + Float32(0.5)) * dth
        var ct = cos(th)
        var stt = sin(th)
        for ip in range(N_PHI):
            var ph = (Float32(ip) + Float32(0.5)) * dph
            var wi = Vec3f(stt * cos(ph), stt * sin(ph), ct)
            var f = coat_eval_smooth(gn, wo, wi, RGB(Float32(1)), IOR)
            acc += f.r * ct * stt * dth * dph
    return acc


comptime N_GRID = 11   # UNIFORM 0.0, 0.1, ..., 1.0 on both axes -- an O(1)
                       # GPU lookup indexes with floor(mu*10)/floor(alpha*10)
                       # and needs no search through unevenly spaced knots.


def main():
    # mu=0.0 is geometrically degenerate (wo exactly grazing); nudge that one
    # grid point to 0.02 so the walk still runs, matching how the LOOKUP
    # will clamp mu to >= a small epsilon before indexing anyway.
    print("mu,alpha,E_walk,E_eval,ratio,corr")
    var s = UInt64(12345)
    for i in range(N_GRID):
        var mu_nom = Float32(i) / Float32(N_GRID - 1)
        var mu = mu_nom if mu_nom > Float32(0.001) else Float32(0.02)
        var ee = eval_albedo(mu)
        var ew0 = Float32(0)
        for j in range(N_GRID):
            var a = Float32(j) / Float32(N_GRID - 1)
            s += 7919
            var ew = walk_albedo(mu, a, s)
            if j == 0: ew0 = ew
            var r = ew / max(ee, Float32(1e-9))
            # corr = E_walk(mu,alpha) / E_walk(mu,0): what lobe_eval's
            # coated_walk branch must be multiplied by so a SMOOTH coat
            # (alpha=0) is bit-unchanged and only the ROUGH excess moves.
            var corr = ew / max(ew0, Float32(1e-9))
            print(String(mu_nom), ",", String(a), ",", String(ew), ",",
                  String(ee), ",", String(r), ",", String(corr), sep="")
