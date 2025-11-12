import Foundation  // atan2

struct HairBsdf: GlobalBsdf {

        init(alpha: FloatX, h: FloatX, absorption: RgbSpectrum, bsdfFrame: BsdfFrame) {
                self.h = h
                self.absorption = absorption
                let sqrtPiOver8: FloatX = 0.626657069
                gammaO = asin(h)
                let azimuthalRoughness: FloatX = 0.3
                s =
                        sqrtPiOver8
                        * (0.265 * azimuthalRoughness + 1.194 * square(azimuthalRoughness) + 5.372
                                * pow(azimuthalRoughness, 2))
                v = Array(repeating: 0, count: pMax + 1)
                let longitudinalRoughness: FloatX = 0.3
                v[0] = square(
                        0.726 * longitudinalRoughness + 0.812 * square(longitudinalRoughness) + 3.7
                                * pow(longitudinalRoughness, 20))
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
                self.bsdfFrame = bsdfFrame
        }

        private func computeAttenutation(
                cosThetaO: FloatX,
                indexRefraction: FloatX,
                h: FloatX,
                transmittance: RgbSpectrum
        ) -> [RgbSpectrum] {
                var attenuation = Array(repeating: black, count: pMax + 1)
                let cosGammaO = (1 - h * h).squareRoot()
                let cosTheta = cosThetaO * cosGammaO
                let fresnelReflected = FresnelDielectric.reflected(
                        cosThetaI: cosTheta,
                        refractiveIndex: indexRefraction)
                let fresnelTransmitted = 1 - fresnelReflected
                attenuation[0] = RgbSpectrum(intensity: fresnelReflected)
                attenuation[1] = square(fresnelTransmitted) * transmittance
                for p in 2..<pMax {
                        attenuation[p] = attenuation[p - 1] * transmittance * fresnelReflected
                }
                let enumerator: RgbSpectrum = attenuation[pMax - 1] * fresnelReflected * transmittance
                let denominator: RgbSpectrum = white - transmittance * fresnelReflected
                attenuation[pMax] = enumerator / denominator
                return attenuation
        }

        private func computeLongitudinalScattering(
                _ cosThetaI: FloatX,
                _ cosThetaO: FloatX,
                _ sinThetaI: FloatX,
                _ sinThetaO: FloatX,
                _ v: FloatX
        ) -> FloatX {
                let a = cosThetaI * cosThetaO / v
                let b = sinThetaI * sinThetaO / v
                var longitudinalScattering: FloatX
                if v <= 0.1 {
                        longitudinalScattering = exp(logI0(a) - b - 1 / v + 0.6931 + log(1 / (2 * v)))
                } else {
                        longitudinalScattering = (exp(-b) * i0(a)) / (sinh(1 / v) * 2 * v)
                }
                return longitudinalScattering
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

        private func computeAzimuthalScattering(
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

        private func computeThetaOp(p: Int, sinThetaO: FloatX, cosThetaO: FloatX) -> (FloatX, FloatX) {
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
                return (sinThetaOp, cosThetaOp)
        }

        private func computeScattering(
                p: Int,
                sinThetaI: FloatX,
                cosThetaI: FloatX,
                sinThetaO: FloatX,
                cosThetaO: FloatX,
                phi: FloatX,
                gammaT: FloatX
        ) -> (FloatX, FloatX) {
                var (sinThetaOp, cosThetaOp) = computeThetaOp(
                        p: p,
                        sinThetaO: sinThetaO,
                        cosThetaO: cosThetaO)
                cosThetaOp = abs(cosThetaOp)
                let longitudinalScattering = computeLongitudinalScattering(
                        cosThetaI,
                        cosThetaOp,
                        sinThetaI,
                        sinThetaOp, v[p])
                let azimuthalScattering = computeAzimuthalScattering(phi, p, s, gammaO, gammaT)
                return (longitudinalScattering, azimuthalScattering)
        }

        func evaluateLocal(wo: Vector, wi: Vector) -> RgbSpectrum {

                let sinThetaO = wo.x
                let cosThetaO = (1 - square(sinThetaO)).squareRoot()
                let phiO = atan2(wo.z, wo.y)

                let sinThetaI = wi.x
                let cosThetaI = (1 - square(sinThetaI)).squareRoot()
                let phiI = atan2(wi.z, wi.y)

                let sinThetaT = sinThetaO / indexRefraction
                let cosThetaT = (1 - square(sinThetaT)).squareRoot()

                let etap = (square(indexRefraction) - square(sinThetaO)).squareRoot() / cosThetaO
                let sinGammaT = h / etap
                let cosGammaT = (1 - square(sinGammaT)).squareRoot()
                let gammaT = asin(sinGammaT)

                // Compute the transmittance _T_ of a single path through the cylinder
                let transmittance = exp(-absorption * (2 * cosGammaT / cosThetaT))

                // Evaluate hair bsdf
                let phi = phiI - phiO
                let attenuation = computeAttenutation(
                        cosThetaO: cosThetaO,
                        indexRefraction: indexRefraction,
                        h: h,
                        transmittance: transmittance)
                var fsum = black

                for p in 0..<pMax {
                        let (longitudinalScattering, azimuthalScattering) = computeScattering(
                                p: p,
                                sinThetaI: sinThetaI,
                                cosThetaI: cosThetaI,
                                sinThetaO: sinThetaO,
                                cosThetaO: cosThetaO,
                                phi: phi,
                                gammaT: gammaT)
                        fsum += longitudinalScattering * attenuation[p] * azimuthalScattering
                }
                let longitudinalScattering = computeLongitudinalScattering(
                        cosThetaI,
                        cosThetaO,
                        sinThetaI,
                        sinThetaO,
                        v[pMax])
                fsum += longitudinalScattering * attenuation[pMax] / (2 * FloatX.pi)
                if absCosTheta(wi) > 0 {
                        fsum /= absCosTheta(wi)
                }
                return fsum
        }

        private func computeAttenuationPdf(cosThetaO: FloatX) -> [FloatX] {
                let sinThetaO = (1 - square(cosThetaO)).squareRoot()
                let sinThetaT = sinThetaO / indexRefraction
                let cosThetaT = (1 - square(sinThetaT)).squareRoot()
                let etap = (square(indexRefraction) - square(sinThetaO)).squareRoot() / cosThetaO
                let sinGammaT = h / etap
                let cosGammaT = (1 - square(sinGammaT)).squareRoot()
                let transmittance = exp(-absorption * (2 * cosGammaT / cosThetaT))
                let attenuation = computeAttenutation(
                        cosThetaO: cosThetaO,
                        indexRefraction: indexRefraction,
                        h: h,
                        transmittance: transmittance)
                let sumY = attenuation.reduce(
                        0,
                        { s, attenuation in
                                s + attenuation.average()
                        })
                return attenuation.map { $0.average() / sumY }
        }

        func probabilityDensityLocal(wo: Vector, wi: Vector) -> FloatX {
                let sinThetaO = wo.x
                let cosThetaO = (1 - square(sinThetaO)).squareRoot()
                let phiO = atan2(wo.z, wo.y)
                let sinThetaI = wi.x
                let cosThetaI = (1 - square(sinThetaI)).squareRoot()
                let phiI = atan2(wi.z, wi.y)
                let etap = sqrt(square(indexRefraction) - square(sinThetaO)) / cosThetaO
                let sinGammaT = h / etap
                let gammaT = asin(sinGammaT)
                let attenuationPdf = computeAttenuationPdf(cosThetaO: cosThetaO)

                let phi = phiI - phiO
                var pdf: FloatX = 0
                for p in 0..<pMax {
                        let (longitudinalScattering, azimuthalScattering) = computeScattering(
                                p: p,
                                sinThetaI: sinThetaI,
                                cosThetaI: cosThetaI,
                                sinThetaO: sinThetaO,
                                cosThetaO: cosThetaO,
                                phi: phi,
                                gammaT: gammaT)
                        pdf += longitudinalScattering * attenuationPdf[p] * azimuthalScattering
                }
                let longitudinalScattering = computeLongitudinalScattering(
                        cosThetaI,
                        cosThetaO,
                        sinThetaI,
                        sinThetaO,
                        v[pMax])
                pdf += longitudinalScattering * attenuationPdf[pMax] * (1 / (2 * FloatX.pi))
                return pdf
        }

        private func compact1by1(_ x: UInt32) -> UInt32 {
                var x = x
                x &= 0x5555_5555
                x = (x ^ (x >> 1)) & 0x3333_3333
                x = (x ^ (x >> 2)) & 0x0f0f_0f0f
                x = (x ^ (x >> 4)) & 0x00ff_00ff
                x = (x ^ (x >> 8)) & 0x0000_ffff
                return x
        }

        private func demux(_ f: FloatX) -> (FloatX, FloatX) {
                let v = UInt(f * FloatX((UInt(1) << 32)))
                let bits: (UInt32, UInt32) = (compact1by1(UInt32(v)), compact1by1(UInt32(v >> 1)))
                return (FloatX(bits.0) / FloatX(1 << 16), FloatX(bits.1) / FloatX(1 << 16))
        }

        private func sampleTrimmedLogistic(u: FloatX, s: FloatX, a: FloatX, b: FloatX) -> FloatX {
                let k = logisticCDF(b, s) - logisticCDF(a, s)
                let x = -s * log(1 / (u * k + logisticCDF(a, s)) - 1)
                return clamp(value: x, low: a, high: b)
        }

        //func sampleLocal(wo: Vector, samples: FourRandomVariables, evaluate: (Vector, Vector) -> RgbSpectrum) async -> (
        //        RgbSpectrum, Vector, FloatX
        //) {
        //        var samples = samples
        //        let sinThetaO = wo.x
        //        let cosThetaO = (1 - square(sinThetaO)).squareRoot()
        //        let phiO = atan2(wo.z, wo.y)

        //        let attenuationPdf = computeAttenuationPdf(cosThetaO: cosThetaO)
        //        var p = 0
        //        for i in 0..<pMax {
        //                p = i
        //                if samples.0 < attenuationPdf[p] {
        //                        break
        //                }
        //                samples.0 -= attenuationPdf[p]
        //        }
        //        let (sinThetaOp, cosThetaOp) = computeThetaOp(
        //                p: p,
        //                sinThetaO: sinThetaO,
        //                cosThetaO: cosThetaO)
        //        samples.1 = max(samples.1, FloatX(1e-5))
        //        let cosTheta = 1 + v[p] * log(samples.1 + (1 - samples.1) * exp(-2 / v[p]))
        //        let sinTheta = (1 - square(cosTheta)).squareRoot()
        //        let cosPhi = cos(2 * FloatX.pi * samples.2)
        //        let sinThetaI = -cosTheta * sinThetaOp + sinTheta * cosPhi * cosThetaOp
        //        let cosThetaI = (1 - square(sinThetaI)).squareRoot()
        //        let etap = (square(indexRefraction) - square(sinThetaO)).squareRoot() / cosThetaO
        //        let sinGammaT = h / etap
        //        let gammaT = asin(sinGammaT)
        //        var dphi: FloatX
        //        if p < pMax {
        //                let phi = computePhi(p: p, gammaO: gammaO, gammaT: gammaT)
        //                let sampled = sampleTrimmedLogistic(
        //                        u: samples.3,
        //                        s: s,
        //                        a: -FloatX.pi,
        //                        b: FloatX.pi)
        //                dphi = phi + sampled
        //        } else {
        //                dphi = 2 * FloatX.pi * samples.3
        //        }
        //        let phiI = phiO + dphi
        //        let wi = Vector3(x: sinThetaI, y: cosThetaI * cos(phiI), z: cosThetaI * sin(phiI))
        //        var pdf: FloatX = 0
        //        for p in 0..<pMax {
        //                let (longitudinalScattering, azimuthalScattering) = computeScattering(
        //                        p: p,
        //                        sinThetaI: sinThetaI,
        //                        cosThetaI: cosThetaI,
        //                        sinThetaO: sinThetaO,
        //                        cosThetaO: cosThetaO,
        //                        phi: dphi,
        //                        gammaT: gammaT)
        //                pdf += longitudinalScattering * attenuationPdf[p] * azimuthalScattering
        //        }
        //        let longitudinalScattering = computeLongitudinalScattering(
        //                cosThetaI,
        //                cosThetaO,
        //                sinThetaI,
        //                sinThetaO,
        //                v[pMax])
        //        pdf += longitudinalScattering * attenuationPdf[pMax] * (1 / (2 * FloatX.pi))
        //        let radiance = evaluate(wo, wi)
        //        return (radiance, wi, pdf)
        //}

        func albedo() -> RgbSpectrum {
                // Not correct but should be ok
                return absorption
        }

        let pMax = 3
        var absorption: RgbSpectrum
        let indexRefraction: FloatX = 1.55
        let gammaO: FloatX
        let h: FloatX
        let s: FloatX
        var v: [FloatX]
        var sin2kAlpha: [FloatX] = Array(repeating: 0, count: 3)
        var cos2kAlpha: [FloatX] = Array(repeating: 0, count: 3)

        let bsdfFrame: BsdfFrame
}
