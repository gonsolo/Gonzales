import Foundation  // atan2

let pMax = 3

struct HairBsdf: BxDF {

        init(alpha: FloatX, h: FloatX, sigmaA: RGBSpectrum) {
                self.h = h
                self.sigmaA = sigmaA

                let sqrtPiOver8: FloatX = 0.626657069

                gammaO = asin(h)

                let betaN: FloatX = 0.3
                s = sqrtPiOver8 * (0.265 * betaN + 1.194 * square(betaN) + 5.372 * pow(betaN, 2))

                let betaM: FloatX = 0.3
                v[0] = square(0.726 * betaM + 0.812 * square(betaM) + 3.7 * pow(betaM, 20))
                v[1] = 0.25 * v[0]
                v[2] = 4 * v[0]
                for p in 3...pMax {
                        v[p] = v[2]
                }

                sin2kAlpha[0] = sin(radians(deg: alpha))
                cos2kAlpha[0] = (1 - square(sin2kAlpha[0])).squareRoot()
                for i in 1..<3 {
                        sin2kAlpha[i] = 2 * cos2kAlpha[i - 1] * sin2kAlpha[i - 1]
                        cos2kAlpha[i] = square(cos2kAlpha[i - 1]) - square(sin2kAlpha[i - 1])
                }
        }

        private func computeAp(
                cosThetaO: FloatX,
                eta: FloatX,
                h: FloatX,
                transmittance: RGBSpectrum
        ) -> [RGBSpectrum] {
                var ap = Array(repeating: black, count: pMax + 1)
                let cosGammaO = (1 - h * h).squareRoot()
                let cosTheta = cosThetaO * cosGammaO
                let f = frDielectric(cosThetaI: cosTheta, etaI: 1, etaT: eta)
                ap[0] = RGBSpectrum(intensity: f)
                ap[1] = square(1 - f) * transmittance
                for p in 2..<pMax {
                        ap[p] = ap[p - 1] * transmittance * f
                }
                ap[pMax] = ap[pMax - 1] * f * transmittance / (white - transmittance * f)
                return ap
        }

        private func computeMp(
                _ cosThetaI: FloatX,
                _ cosThetaO: FloatX,
                _ sinThetaI: FloatX,
                _ sinThetaO: FloatX,
                _ v: FloatX
        ) -> FloatX {
                let a = cosThetaI * cosThetaO / v
                let b = sinThetaI * sinThetaO / v
                var mp: FloatX
                if v <= 0.1 {
                        mp = exp(logI0(a) - b - 1 / v + 0.6931 + log(1 / (2 * v)))
                } else {
                        mp = (exp(-b) * i0(a)) / (sinh(1 / v) * 2 * v)
                }
                return mp
        }

        private func computePhi(p: Int, gammaO: FloatX, gammaT: FloatX) -> FloatX {
                return 2 * FloatX(p) * gammaT - 2 * gammaO + FloatX(p) * FloatX.pi
        }

        private func logistic(_ x: FloatX, _ s: FloatX) -> FloatX {
                let x = abs(x)
                return exp(-x / s) / (s * square(1 + exp(-x / s)))
        }

        private func logisticCDF(_ x: FloatX, _ s: FloatX) -> FloatX {
                return 1 / (1 + exp(-x / s))
        }

        private func trimmedLogistic(_ x: FloatX, _ s: FloatX, _ a: FloatX, _ b: FloatX) -> FloatX {
                return logistic(x, s) / (logisticCDF(b, s) - logisticCDF(a, s))
        }

        private func computeNp(
                _ phi: FloatX,
                _ p: Int,
                _ s: FloatX,
                _ gammaO: FloatX,
                _ gammaT: FloatX
        ) -> FloatX {
                var dphi = phi - computePhi(p: p, gammaO: gammaO, gammaT: gammaT)
                while dphi > FloatX.pi {
                        dphi -= 2 * FloatX.pi
                }
                while dphi < -FloatX.pi {
                        dphi += 2 * FloatX.pi
                }
                return trimmedLogistic(dphi, s, -FloatX.pi, FloatX.pi)
        }

        private func i0(_ x: FloatX) -> FloatX {
                var val: FloatX = 0
                var x2i: FloatX = 1
                var ifact: FloatX = 1
                var i4: FloatX = 1
                for i in 0..<10 {
                        if i > 1 {
                                ifact *= FloatX(i)
                        }
                        val += x2i / (i4 * square(ifact))
                        x2i *= x * x
                        i4 *= 4
                }
                return val
        }

        private func logI0(_ x: FloatX) -> FloatX {
                if x > 12 {
                        return x + 0.5 * (-log(2 * FloatX.pi) + log(1 / x) + 1 / (8 * x))
                } else {
                        return log(i0(x))
                }
        }

        func evaluate(wo: Vector, wi: Vector) -> RGBSpectrum {

                let sinThetaO = wo.x
                let cosThetaO = (1 - square(sinThetaO)).squareRoot()
                let phiO = atan2(wo.z, wo.y)

                let sinThetaI = wi.x
                let cosThetaI = (1 - square(sinThetaI)).squareRoot()
                let phiI = atan2(wi.z, wi.y)

                let sinThetaT = sinThetaO / eta
                let cosThetaT = (1 - square(sinThetaT)).squareRoot()

                let etap = (eta * eta - square(sinThetaO)).squareRoot() / cosThetaO
                let sinGammaT = h / etap
                let cosGammaT = (1 - square(sinGammaT)).squareRoot()
                let gammaT = asin(sinGammaT)

                // Compute the transmittance _T_ of a single path through the cylinder
                let transmittance = exp(-sigmaA * (2 * cosGammaT / cosThetaT))

                // Evaluate hair BSDF
                let phi = phiI - phiO
                let ap = computeAp(
                        cosThetaO: cosThetaO,
                        eta: eta,
                        h: h,
                        transmittance: transmittance)
                var fsum = black

                for p in 0..<pMax {
                        var sinThetaOp: FloatX
                        var cosThetaOp: FloatX
                        if p == 0 {
                                sinThetaOp = sinThetaO * cos2kAlpha[1] - cosThetaO * sin2kAlpha[1]
                                cosThetaOp = cosThetaO * cos2kAlpha[1] + sinThetaO * sin2kAlpha[1]
                        } else if p == 1 {
                                sinThetaOp = sinThetaO * cos2kAlpha[0] + cosThetaO * sin2kAlpha[0]
                                cosThetaOp = cosThetaO * cos2kAlpha[0] - sinThetaO * sin2kAlpha[0]
                        } else if p == 2 {
                                sinThetaOp = sinThetaO * cos2kAlpha[2] + cosThetaO * sin2kAlpha[2]
                                cosThetaOp = cosThetaO * cos2kAlpha[2] - sinThetaO * sin2kAlpha[2]
                        } else {
                                sinThetaOp = sinThetaO
                                cosThetaOp = cosThetaO
                        }
                        cosThetaOp = abs(cosThetaOp)

                        let mp = computeMp(cosThetaI, cosThetaOp, sinThetaI, sinThetaOp, v[p])
                        let np = computeNp(phi, p, s, gammaO, gammaT)
                        fsum += mp * ap[p] * np
                }
                let mp = computeMp(cosThetaI, cosThetaO, sinThetaI, sinThetaO, v[pMax])
                fsum += mp * ap[pMax] / (2 * FloatX.pi)
                if absCosTheta(wi) > 0 {
                        fsum /= absCosTheta(wi)
                }
                return fsum
        }

        private func computeApPdf(cosThetaO: FloatX) -> [FloatX] {
                let sinThetaO = (1 - cosThetaO * cosThetaO).squareRoot()
                let sinThetaT = sinThetaO / eta
                let cosThetaT = (1 - square(sinThetaT)).squareRoot()
                let etap = (eta * eta - square(sinThetaO)).squareRoot() / cosThetaO
                let sinGammaT = h / etap
                let cosGammaT = (1 - square(sinGammaT)).squareRoot()
                let transmittance = exp(-sigmaA * (2 * cosGammaT / cosThetaT))
                let ap = computeAp(
                        cosThetaO: cosThetaO,
                        eta: eta,
                        h: h,
                        transmittance: transmittance)
                let sumY = ap.reduce(
                        0,
                        { s, ap in
                                s + ap.average()
                        })
                let apPdf = ap.map { $0.average() / sumY }
                return apPdf
        }

        func probabilityDensity(wo: Vector, wi: Vector) -> FloatX {
                let sinThetaO = wo.x
                let cosThetaO = (1 - square(sinThetaO)).squareRoot()
                let phiO = atan2(wo.z, wo.y)
                let sinThetaI = wi.x
                let cosThetaI = (1 - square(sinThetaI)).squareRoot()
                let phiI = atan2(wi.z, wi.y)
                let etap = sqrt(eta * eta - square(sinThetaO)) / cosThetaO
                let sinGammaT = h / etap
                let gammaT = asin(sinGammaT)
                let apPdf = computeApPdf(cosThetaO: cosThetaO)

                let phi = phiI - phiO
                var pdf: FloatX = 0
                for p in 0..<pMax {
                        var sinThetaOp: FloatX
                        var cosThetaOp: FloatX
                        if p == 0 {
                                sinThetaOp = sinThetaO * cos2kAlpha[1] - cosThetaO * sin2kAlpha[1]
                                cosThetaOp = cosThetaO * cos2kAlpha[1] + sinThetaO * sin2kAlpha[1]
                        } else if p == 1 {
                                sinThetaOp = sinThetaO * cos2kAlpha[0] + cosThetaO * sin2kAlpha[0]
                                cosThetaOp = cosThetaO * cos2kAlpha[0] - sinThetaO * sin2kAlpha[0]
                        } else if p == 2 {
                                sinThetaOp = sinThetaO * cos2kAlpha[2] + cosThetaO * sin2kAlpha[2]
                                cosThetaOp = cosThetaO * cos2kAlpha[2] - sinThetaO * sin2kAlpha[2]
                        } else {
                                sinThetaOp = sinThetaO
                                cosThetaOp = cosThetaO
                        }
                        cosThetaOp = abs(cosThetaOp)
                        let mp = computeMp(cosThetaI, cosThetaOp, sinThetaI, sinThetaOp, v[p])
                        let np = computeNp(phi, p, s, gammaO, gammaT)
                        pdf += mp * apPdf[p] * np
                }
                let mp = computeMp(cosThetaI, cosThetaO, sinThetaI, sinThetaO, v[pMax])
                pdf += mp * apPdf[pMax] * (1 / (2 * FloatX.pi))
                return pdf
        }

        //private func demux(_ f: FloatX) -> (FloatX, FloatX) {
        //        let v = f * (UInt(1) << 32)
        //        // TODOuint32_t bits[2] = {Compact1By1(v), Compact1By1(v >> 1)};
        //        // TODOreturn {bits[0] / Float(1 << 16), bits[1] / Float(1 << 16)};
        //        return (0, 0)
        //}

        //private func demux(_ u: Point2F) -> (FloatX, FloatX, FloatX, FloatX) {
        //        let a = demux(u[0])
        //        let b = demus(u[1])
        //        return (a, b)
        //}

        //func sample(wo: Vector, u: Point2F, evaluate: (Vector, Vector) -> RGBSpectrum) -> (
        //        RGBSpectrum, Vector, FloatX
        //) {
        //        let sinThetaO = wo.x
        //        let cosThetaO = (1 - square(sinThetaO)).squareRoot()
        //        let phiO = atan2(wo.z, wo.y)
        //        // TODO let fourU = demux(u)

        //        //    Point2f u[2] = {DemuxFloat(u2[0]), DemuxFloat(u2[1])};
        //        //
        //        //    // Determine which term $p$ to sample for hair scattering
        //        //    std::array<Float, pMax + 1> apPdf = ComputeApPdf(cosThetaO);
        //        //    int p;
        //        //    for (p = 0; p < pMax; ++p) {
        //        //        if (u[0][0] < apPdf[p]) break;
        //        //        u[0][0] -= apPdf[p];
        //        //    }
        //        //
        //        //    // Rotate $\sin \thetao$ and $\cos \thetao$ to account for hair scale tilt
        //        //    Float sinThetaOp, cosThetaOp;
        //        //    if (p == 0) {
        //        //        sinThetaOp = sinThetaO * cos2kAlpha[1] - cosThetaO * sin2kAlpha[1];
        //        //        cosThetaOp = cosThetaO * cos2kAlpha[1] + sinThetaO * sin2kAlpha[1];
        //        //    }
        //        //    else if (p == 1) {
        //        //        sinThetaOp = sinThetaO * cos2kAlpha[0] + cosThetaO * sin2kAlpha[0];
        //        //        cosThetaOp = cosThetaO * cos2kAlpha[0] - sinThetaO * sin2kAlpha[0];
        //        //    } else if (p == 2) {
        //        //        sinThetaOp = sinThetaO * cos2kAlpha[2] + cosThetaO * sin2kAlpha[2];
        //        //        cosThetaOp = cosThetaO * cos2kAlpha[2] - sinThetaO * sin2kAlpha[2];
        //        //    } else {
        //        //        sinThetaOp = sinThetaO;
        //        //        cosThetaOp = cosThetaO;
        //        //    }
        //        //
        //        //   // Sample $M_p$ to compute $\thetai$
        //        //    u[1][0] = std::max(u[1][0], Float(1e-5));
        //        //    Float cosTheta =
        //        //        1 + v[p] * std::log(u[1][0] + (1 - u[1][0]) * std::exp(-2 / v[p]));
        //        //    Float sinTheta = SafeSqrt(1 - Sqr(cosTheta));
        //        //    Float cosPhi = std::cos(2 * Pi * u[1][1]);
        //        //    Float sinThetaI = -cosTheta * sinThetaOp + sinTheta * cosPhi * cosThetaOp;
        //        //    Float cosThetaI = SafeSqrt(1 - Sqr(sinThetaI));
        //        //
        //        //    // Sample $N_p$ to compute $\Delta\phi$
        //        //
        //        //    // Compute $\gammat$ for refracted ray
        //        //    Float etap = std::sqrt(eta * eta - Sqr(sinThetaO)) / cosThetaO;
        //        //    Float sinGammaT = h / etap;
        //        //    Float gammaT = SafeASin(sinGammaT);
        //        //    Float dphi;
        //        //    if (p < pMax)
        //        //        dphi =
        //        //            Phi(p, gammaO, gammaT) + SampleTrimmedLogistic(u[0][1], s, -Pi, Pi);
        //        //    else
        //        //        dphi = 2 * Pi * u[0][1];
        //        //
        //        //    // Compute _wi_ from sampled hair scattering angles
        //        //    Float phiI = phiO + dphi;
        //        //    *wi = Vector3f(sinThetaI, cosThetaI * std::cos(phiI),
        //        //                   cosThetaI * std::sin(phiI));
        //        //
        //        //
        //        //   // Compute PDF for sampled hair scattering direction _wi_
        //        //    *pdf = 0;
        //        //    for (int p = 0; p < pMax; ++p) {
        //        //        // Compute $\sin \thetao$ and $\cos \thetao$ terms accounting for scales
        //        //        Float sinThetaOp, cosThetaOp;
        //        //        if (p == 0) {
        //        //            sinThetaOp = sinThetaO * cos2kAlpha[1] - cosThetaO * sin2kAlpha[1];
        //        //            cosThetaOp = cosThetaO * cos2kAlpha[1] + sinThetaO * sin2kAlpha[1];
        //        //        }
        //        //
        //        //        // Handle remainder of $p$ values for hair scale tilt
        //        //        else if (p == 1) {
        //        //            sinThetaOp = sinThetaO * cos2kAlpha[0] + cosThetaO * sin2kAlpha[0];
        //        //            cosThetaOp = cosThetaO * cos2kAlpha[0] - sinThetaO * sin2kAlpha[0];
        //        //        } else if (p == 2) {
        //        //            sinThetaOp = sinThetaO * cos2kAlpha[2] + cosThetaO * sin2kAlpha[2];
        //        //            cosThetaOp = cosThetaO * cos2kAlpha[2] - sinThetaO * sin2kAlpha[2];
        //        //        } else {
        //        //            sinThetaOp = sinThetaO;
        //        //            cosThetaOp = cosThetaO;
        //        //        }
        //        //
        //        //        // Handle out-of-range $\cos \thetao$ from scale adjustment
        //        //        cosThetaOp = std::abs(cosThetaOp);
        //        //        *pdf += Mp(cosThetaI, cosThetaOp, sinThetaI, sinThetaOp, v[p]) *
        //        //                apPdf[p] * Np(dphi, p, s, gammaO, gammaT);
        //        //    }
        //        //    *pdf += Mp(cosThetaI, cosThetaO, sinThetaI, sinThetaO, v[pMax]) *
        //        //            apPdf[pMax] * (1 / (2 * Pi));
        //        //    // if (std::abs(wi->x) < .9999) CHECK_NEAR(*pdf, Pdf(wo, *wi), .01);
        //        //    return f(wo, *wi);

        //        return (black, up, 0)
        //}

        func albedo() -> RGBSpectrum {
                // TODO
                return white
        }

        var sigmaA: RGBSpectrum
        let eta: FloatX = 1.55
        let gammaO: FloatX
        let h: FloatX
        let s: FloatX
        var v: [FloatX] = Array(repeating: 0, count: pMax + 1)
        var sin2kAlpha: [FloatX] = Array(repeating: 0, count: 3)
        var cos2kAlpha: [FloatX] = Array(repeating: 0, count: 3)
}
