"""CAMIS's merge-weight evaluator, pinned to the exact values derived in
Scenes/vcm_camis_hybrid_derivation.py's section_pinned().

That harness builds trace-blind camera/light records for one toy path (the
same path as Scenes/vcm_camis_derivation.py's scene(): z0 -> z1 -> x -> y2 ->
y1 -> lens, merged at x) and reconstructs Eq. 11's exact weight from them
alone, agreeing with the direct post-hoc computation to ~1e-12 in double
precision. This test pins camis_eval_merge -- which internally exercises
camis_c and both camis_camera_side/camis_light_side's Horner-form record
walks -- to that harness's printed table, so a formula regression anywhere
in that chain is caught in milliseconds instead of as a percent-level
brightness shift once S3 wires the real weight sites up (this file only
tests the evaluators; nothing here is wired into bdpt.mojo yet, per
vcm_camis.mojo's stage S1/S2 split).

Tolerance is 1e-4 relative: the harness's own Float32 emulation (section 4)
measured a worst-case partition error of 1.6e-7 for the Horner form these
functions use, on top of which this comparison also crosses double (the
harness) vs. Float32 (here) rounding -- both comfortably inside 1e-4.
"""
from std.testing import assert_true, TestSuite
from std.collections import Array
from gonzales.vcm_camis import (
    CamisCamCarry, CamisCamRecord, CamisLightRecord, CAMIS_IN_CLASS,
    camis_eval_merge,
)

# ── the harness's toy path a, merge at x (camera y1 y2 x; light z0 z1 x) ───
comptime N_T = Float32(20000.0)
comptime LOG_PI_R2 = Float32(-4.148160570610315)

# Camera records tau = 0 (y1), 1 (y2); Horner form (a, eta*b) + {log_keep, log_py, rc}.
comptime CAM0_LOG_KEEP = Float32(0.0)
comptime CAM0_LOG_PY = Float32(0.0)
comptime CAM0_RC = Float32(0.0)
comptime CAM0_A = Float32(1.3131555206909384)
comptime CAM0_ETAB = Float32(236.71356485207008)
comptime CAM1_LOG_KEEP = Float32(0.0)
comptime CAM1_LOG_PY = Float32(-7.181127882797374)
comptime CAM1_RC = Float32(-7.181127882797374)
comptime CAM1_A = Float32(1.1050172874838042)
comptime CAM1_ETAB = Float32(261.572581343456)

# Camera running registers on arrival at x (tau = 2); cut = 0.
comptime CAM_DVCM = Float32(21.854631002056042)
comptime CAM_DVC_LEGACY = Float32(696.6113080012652)
comptime CAM_DVC0 = Float32(173.46614531435296)
comptime CAM_LOG_PY = Float32(-14.413701297666272)
comptime CAM_RC_PREV = Float32(-7.181127882797374)
comptime CAM_LOG_G_PREV = Float32(-1.552802685292427)
comptime CAM_ETA = Float32(56.548667764616276)
comptime CAM_LOG_KEEP = Float32(0.0)

# Light: origin, one stored vertex lam = 1 (z1), arrival at x (lam = 2).
comptime LIGHT_LOG_PA0 = Float32(1.3862943611198906)
comptime LIGHT1_LOG_KEEP = Float32(0.0)
comptime LIGHT1_LOG_PA_FWD = Float32(-2.6582867186869334)
comptime LIGHT1_LOG_PA_REV = Float32(-2.6582867186869334)
comptime LIGHT1_A = Float32(0.8649617257741229)
comptime LIGHT1_ETAB = Float32(226.2692034594269)

# Light arrival registers; cut = 1.
comptime LIGHT_DVCM = Float32(9.547905123891352)
comptime LIGHT_DVC_LEGACY = Float32(284.37560955203844)
comptime LIGHT_DVC0 = Float32(58.10640609261156)
comptime LIGHT_LOG_PA_FWD = Float32(-2.256321771675156)
comptime LIGHT_LOG_G_REV = Float32(-0.8696979592738109)
comptime LIGHT_ETA = Float32(56.548667764616276)

# Junction (evaluated at the merge site).
comptime CAM_DIR_W = Float32(0.24991765074541492)
comptime CAM_REV_W = Float32(0.2161872910156644)

# Expected merge weight, CAMIS on and off (Eq. 11's balance heuristic).
comptime EXPECT_MERGE_W_CAMIS = Float32(0.19207528187507197)
comptime EXPECT_MERGE_W_LEGACY = Float32(0.18263881947791993)

comptime TOL = Float32(1e-4)


def _close(got: Float32, want: Float32, label: String) raises:
    var err = abs(got - want) / max(abs(want), Float32(1e-30))
    if err > TOL:
        raise Error(String("{}: got {}, want {}, rel err {}").format(label, got, want, err))


def _cam_carry() -> CamisCamCarry:
    return CamisCamCarry(True, Int32(0), LOG_PI_R2, CAM_LOG_PY, CAM_RC_PREV,
                         CAM_LOG_G_PREV, Float32(0))


def _cam_scat() -> Array[CamisCamRecord, 2]:
    var recs = Array[CamisCamRecord, 2](fill=CamisCamRecord(
        Float32(0), Float32(0), Float32(0), Float32(0), Float32(0)))
    recs[0] = CamisCamRecord(CAM0_A, CAM0_ETAB, CAM0_LOG_KEEP, CAM0_LOG_PY, CAM0_RC)
    recs[1] = CamisCamRecord(CAM1_A, CAM1_ETAB, CAM1_LOG_KEEP, CAM1_LOG_PY, CAM1_RC)
    return recs^


def _light_arr() -> CamisLightRecord:
    return CamisLightRecord(Float32(0), Float32(0), Float32(0), Float32(0),
                            LIGHT_LOG_G_REV, Int32(1), CAMIS_IN_CLASS)


def _light_scat() -> Array[CamisLightRecord, 1]:
    var recs = Array[CamisLightRecord, 1](fill=CamisLightRecord(
        Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Int32(0), Int32(0)))
    recs[0] = CamisLightRecord(LIGHT1_A, LIGHT1_ETAB, LIGHT1_LOG_KEEP,
                               LIGHT1_LOG_PA_REV, Float32(0), Int32(1), CAMIS_IN_CLASS)
    return recs^


def _light_log_pa_fwd() -> Array[Float32, 1]:
    var a = Array[Float32, 1](fill=Float32(0))
    a[0] = LIGHT1_LOG_PA_FWD
    return a^


def test_camis_eval_merge_matches_the_pinned_weight() raises:
    var w = camis_eval_merge[2, 1](
        _cam_carry(), _cam_scat(), 2,
        CAM_DVCM, CAM_DVC_LEGACY, CAM_DVC0, CAM_ETA, CAM_LOG_KEEP,
        _light_arr(), _light_scat(), 1,
        _light_log_pa_fwd(), LIGHT_LOG_PA_FWD,
        LIGHT_LOG_PA0, True,
        LIGHT_DVCM, LIGHT_DVC_LEGACY, LIGHT_DVC0, LIGHT_ETA,
        CAM_DIR_W, CAM_REV_W,
        N_T, True,
    )
    _close(w, EXPECT_MERGE_W_CAMIS, "merge weight, CAMIS on")


def test_camis_eval_merge_off_matches_the_legacy_weight() raises:
    """CAMIS_ON = False takes the exact same branch _bdpt_merge_mis_weight's
    existing (non-CAMIS) code does today -- this is the "no regression on
    the default build" identity the plan's G1 formalizes."""
    var w = camis_eval_merge[2, 1](
        _cam_carry(), _cam_scat(), 2,
        CAM_DVCM, CAM_DVC_LEGACY, CAM_DVC0, CAM_ETA, CAM_LOG_KEEP,
        _light_arr(), _light_scat(), 1,
        _light_log_pa_fwd(), LIGHT_LOG_PA_FWD,
        LIGHT_LOG_PA0, True,
        LIGHT_DVCM, LIGHT_DVC_LEGACY, LIGHT_DVC0, LIGHT_ETA,
        CAM_DIR_W, CAM_REV_W,
        N_T, False,
    )
    _close(w, EXPECT_MERGE_W_LEGACY, "merge weight, CAMIS off")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
