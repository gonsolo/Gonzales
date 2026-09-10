#!/usr/bin/env python3
"""Phase 9.2 measurement: is the naive-plug-in bias bound usable at gonzales's
real scales?  (Answer: no -- see docs/A2_restir_migration_plan.md Phase 9.2.)

Phase 9's joint balance heuristic needs SMS's density q in the DENOMINATOR
    F = n_SMS * q_SMS + C
where C is the other techniques' densities.  Substituting a noisy estimate
q_hat biases the weight.  The recorded proposal was to use the naive plug-in
wherever a closed-form bias bound is small, and fall back to Phase 6's
separate reservoir otherwise.

ALGEBRA (second-order Taylor; delta = n*(q_hat-q), E[delta]=0):
    E[p/F] ~= (p/F0) * (1 + Var(delta)/F0^2)
    relative bias = n^2 Var(q_hat) / F0^2
With a hit-rate estimator over M seeds, Var(q_hat) = q(1-q)/M exactly, and
writing r = C/(n*q):

    relative bias = (1 - q) / ( M * q * (1 + r)^2 )        <-- n_SMS CANCELS

So the bias is governed by q and M, not by the n_i scales the plan expected.

HOW q WAS MEASURED.  gonzales's `sms_solve_bernoulli` returns a trial count
T ~ Geometric(q) with E[T] = 1/q -- unbiased for the RECIPROCAL.  Phase 9.2
needs q itself, and 1/T is not an unbiased estimator of it.  So sms.mojo
carries a diagnostic hit-rate probe (`SMS_QHAT_PROBE_M`, default 0 = off):
fire M independent fresh seeds, count the fraction converging back to X*.
Set it to 64 and render to regenerate the input:

    ./build/gonzales --spp 16 --resolution 200x200 \
        <.../Figure_14_15_MainComparison/sphere_sms.xml> 2>&1 \
        | grep '^QSMS' > qsms.txt

sphere_sms.xml is the scene where SMS demonstrably works (matches a
brute-force ground truth to 2.4%).  water-caustic is NOT usable: SMS
produces no caustic there at all in batch mode.
"""
import sys
import numpy as np

M_DEFAULT = 64


def load(path):
    rows = [l.split() for l in open(path) if l.startswith("QSMS")]
    k = np.array([int(r[1]) for r in rows], dtype=float)
    M = int(rows[0][2])
    return k / M, M


def rel_bias(q, M, r):
    """Closed-form relative bias of the naive plug-in."""
    return np.where(q > 0, (1.0 - q) / (M * np.maximum(q, 1e-12) * (1.0 + r) ** 2), np.inf)


def main(path):
    q, M = load(path)
    print(f"N = {len(q)} SMS solves,  M = {M}\n")
    print("q quantiles:", "  ".join(f"p{p}={np.percentile(q, p):.3f}"
                                    for p in (1, 5, 25, 50, 75, 95)))
    print(f"mean q = {q.mean():.3f}   q==0: {(q == 0).mean():.2%}\n")

    print(f"{'r':>5} {'median bias':>12} {'p95':>11} {'%>1% (fallback)':>17}")
    for r in (0.0, 1.0, 4.0, 10.0):
        b = rel_bias(q, M, r)
        fin = np.isfinite(b)
        print(f"{r:>5g} {np.median(b[fin]):>12.2e} {np.percentile(b[fin], 95):>11.2e}"
              f" {(b >= 0.01).mean():>16.1%}")

    print("\nExtra Newton solves M needed for 1% bias at r=0 "
          "(existing Bernoulli estimator: median T=1):")
    qp = q[q > 0]
    for p in (50, 25, 5):
        qq = np.percentile(qp, p)
        print(f"   at p{p} q={qq:.3f}:  M >= {int(np.ceil((1 - qq) / (0.01 * qq)))}")

    print("\nP(q_hat == 0), which makes F = 0 when C = 0 -- no M fixes the tail:")
    for m in (64, 256, 1024):
        print(f"   M={m:<5} {np.mean((1 - q) ** m):.2e}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "qsms.txt")
