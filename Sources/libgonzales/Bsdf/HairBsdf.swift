import Foundation  // atan2

struct HairBsdf: GlobalBsdf {

        let pMax = 3
        var absorption: RgbSpectrum
        let indexRefraction: FloatX = 1.55
        let gammaO: FloatX
        let hOffset: FloatX
        let sParameter: FloatX
        var vParameter: [FloatX]
        var sin2kAlpha: [FloatX] = Array(repeating: 0, count: 3)
        var cos2kAlpha: [FloatX] = Array(repeating: 0, count: 3)
        let bsdfFrame: BsdfFrame

        init(alpha: FloatX, hOffset: FloatX, absorption: RgbSpectrum, bsdfFrame: BsdfFrame) {
                self.hOffset = hOffset
                self.absorption = absorption
                let sqrtPiOver8: FloatX = 0.626657069
                gammaO = asin(hOffset)
                let azimuthalRoughness: FloatX = 0.3
                sParameter =
                        sqrtPiOver8
                        * (0.265 * azimuthalRoughness + 1.194 * square(azimuthalRoughness) + 5.372
                                * pow(azimuthalRoughness, 2))
                vParameter = Array(repeating: 0, count: pMax + 1)
                let longitudinalRoughness: FloatX = 0.3
                vParameter[0] = square(
                        0.726 * longitudinalRoughness + 0.812 * square(longitudinalRoughness) + 3.7
                                * pow(longitudinalRoughness, 20))
                vParameter[1] = 0.25 * vParameter[0]
                vParameter[2] = 4 * vParameter[0]
                for pIndex in 3...pMax {
                        vParameter[pIndex] = vParameter[2]
                }
                sin2kAlpha[0] = sin(radians(deg: alpha))
                cos2kAlpha[0] = (1 - square(sin2kAlpha[0])).squareRoot()
                for index in 1..<3 {
                        sin2kAlpha[index] = 2 * cos2kAlpha[index - 1] * sin2kAlpha[index - 1]
                        cos2kAlpha[index] = square(cos2kAlpha[index - 1]) - square(sin2kAlpha[index - 1])
                }
                self.bsdfFrame = bsdfFrame
        }
}

extension HairBsdf {

        private func computeAttenutation(
                cosThetaO: FloatX,
                indexRefraction: FloatX,
                hOffset: FloatX,
                transmittance: RgbSpectrum
        ) -> [RgbSpectrum] {
                var attenuation = Array(repeating: black, count: pMax + 1)
                let cosGammaO = (1 - hOffset * hOffset).squareRoot()
                let cosTheta = cosThetaO * cosGammaO
                let fresnelReflected = FresnelDielectric.reflected(
                        cosThetaI: cosTheta,
                        refractiveIndex: indexRefraction)
                let fresnelTransmitted = 1 - fresnelReflected
                attenuation[0] = RgbSpectrum(intensity: fresnelReflected)
                attenuation[1] = square(fresnelTransmitted) * transmittance
                for pIndex in 2..<pMax {
                        attenuation[pIndex] = attenuation[pIndex - 1] * transmittance * fresnelReflected
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
                _ vParameter: FloatX
        ) -> FloatX {
                let termA = cosThetaI * cosThetaO / vParameter
                let termB = sinThetaI * sinThetaO / vParameter
                var longitudinalScattering: FloatX
                if vParameter <= 0.1 {
                        longitudinalScattering = exp(
                                logI0(termA) - termB - 1 / vParameter + 0.6931 + log(1 / (2 * vParameter)))
                } else {
                        longitudinalScattering = (exp(-termB) * i0(termA)) / (sinh(1 / vParameter) * 2 * vParameter)
                }
                return longitudinalScattering
        }

        private func computePhi(pIndex: Int, gammaO: FloatX, gammaT: FloatX) -> FloatX {
                return 2 * FloatX(pIndex) * gammaT - 2 * gammaO + FloatX(pIndex) * FloatX.pi
        }

        private func logistic(_ xVal: FloatX, _ sVal: FloatX) -> FloatX {
                let xVal = abs(xVal)
                return exp(-xVal / sVal) / (sVal * square(1 + exp(-xVal / sVal)))
        }

        private func logisticCDF(_ xVal: FloatX, _ sVal: FloatX) -> FloatX {
                return 1 / (1 + exp(-xVal / sVal))
        }

        private func trimmedLogistic(_ xVal: FloatX, _ sVal: FloatX, _ lower: FloatX, _ upper: FloatX) -> FloatX {
                return logistic(xVal, sVal) / (logisticCDF(upper, sVal) - logisticCDF(lower, sVal))
        }

        private func computeAzimuthalScattering(
                _ phi: FloatX,
                _ pIndex: Int,
                _ sVal: FloatX,
                _ gammaO: FloatX,
                _ gammaT: FloatX
        ) -> FloatX {
                var dphi = phi - computePhi(pIndex: pIndex, gammaO: gammaO, gammaT: gammaT)
                while dphi > FloatX.pi {
                        dphi -= 2 * FloatX.pi
                }
                while dphi < -FloatX.pi {
                        dphi += 2 * FloatX.pi
                }
                return trimmedLogistic(dphi, sVal, -FloatX.pi, FloatX.pi)
        }

        private func i0(_ xVal: FloatX) -> FloatX {
                var val: FloatX = 0
                var x2i: FloatX = 1
                var ifact: FloatX = 1
                var powerOfFour: FloatX = 1
                for index in 0..<10 {
                        if index > 1 {
                                ifact *= FloatX(index)
                        }
                        val += x2i / (powerOfFour * square(ifact))
                        x2i *= xVal * xVal
                        powerOfFour *= 4
                }
                return val
        }

        private func logI0(_ xVal: FloatX) -> FloatX {
                if xVal > 12 {
                        return xVal + 0.5 * (-log(2 * FloatX.pi) + log(1 / xVal) + 1 / (8 * xVal))
                } else {
                        return log(i0(xVal))
                }
        }

        private func computeThetaOp(pIndex: Int, sinThetaO: FloatX, cosThetaO: FloatX) -> (FloatX, FloatX) {
                var sinThetaOp: FloatX
                var cosThetaOp: FloatX
                if pIndex == 0 {
                        sinThetaOp = sinThetaO * cos2kAlpha[1] - cosThetaO * sin2kAlpha[1]
                        cosThetaOp = cosThetaO * cos2kAlpha[1] + sinThetaO * sin2kAlpha[1]
                } else if pIndex == 1 {
                        sinThetaOp = sinThetaO * cos2kAlpha[0] + cosThetaO * sin2kAlpha[0]
                        cosThetaOp = cosThetaO * cos2kAlpha[0] - sinThetaO * sin2kAlpha[0]
                } else if pIndex == 2 {
                        sinThetaOp = sinThetaO * cos2kAlpha[2] + cosThetaO * sin2kAlpha[2]
                        cosThetaOp = cosThetaO * cos2kAlpha[2] - sinThetaO * sin2kAlpha[2]
                } else {
                        sinThetaOp = sinThetaO
                        cosThetaOp = cosThetaO
                }
                return (sinThetaOp, cosThetaOp)
        }

        private func computeScattering(
                pIndex: Int,
                sinThetaI: FloatX,
                cosThetaI: FloatX,
                sinThetaO: FloatX,
                cosThetaO: FloatX,
                phi: FloatX,
                gammaT: FloatX
        ) -> (FloatX, FloatX) {
                var (sinThetaOp, cosThetaOp) = computeThetaOp(
                        pIndex: pIndex,
                        sinThetaO: sinThetaO,
                        cosThetaO: cosThetaO)
                cosThetaOp = abs(cosThetaOp)
                let longitudinalScattering = computeLongitudinalScattering(
                        cosThetaI,
                        cosThetaOp,
                        sinThetaI,
                        sinThetaOp, vParameter[pIndex])
                let azimuthalScattering = computeAzimuthalScattering(phi, pIndex, sParameter, gammaO, gammaT)
                return (longitudinalScattering, azimuthalScattering)
        }

        func evaluateLocal(outgoing: Vector, incident: Vector) -> RgbSpectrum {

                let sinThetaO = outgoing.x
                let cosThetaO = (1 - square(sinThetaO)).squareRoot()
                let phiO = atan2(outgoing.z, outgoing.y)

                let sinThetaI = incident.x
                let cosThetaI = (1 - square(sinThetaI)).squareRoot()
                let phiI = atan2(incident.z, incident.y)

                let sinThetaT = sinThetaO / indexRefraction
                let cosThetaT = (1 - square(sinThetaT)).squareRoot()

                let etap = (square(indexRefraction) - square(sinThetaO)).squareRoot() / cosThetaO
                let sinGammaT = hOffset / etap
                let cosGammaT = (1 - square(sinGammaT)).squareRoot()
                let gammaT = asin(sinGammaT)

                // Compute the transmittance _T_ of a single path through the cylinder
                let transmittance = exp(-absorption * (2 * cosGammaT / cosThetaT))

                // Evaluate hair bsdf
                let phi = phiI - phiO
                let attenuation = computeAttenutation(
                        cosThetaO: cosThetaO,
                        indexRefraction: indexRefraction,
                        hOffset: hOffset,
                        transmittance: transmittance)
                var fsum = black

                for pIndex in 0..<pMax {
                        let (longitudinalScattering, azimuthalScattering) = computeScattering(
                                pIndex: pIndex,
                                sinThetaI: sinThetaI,
                                cosThetaI: cosThetaI,
                                sinThetaO: sinThetaO,
                                cosThetaO: cosThetaO,
                                phi: phi,
                                gammaT: gammaT)
                        fsum += longitudinalScattering * attenuation[pIndex] * azimuthalScattering
                }
                let longitudinalScattering = computeLongitudinalScattering(
                        cosThetaI,
                        cosThetaO,
                        sinThetaI,
                        sinThetaO,
                        vParameter[pMax])
                fsum += longitudinalScattering * attenuation[pMax] / (2 * FloatX.pi)
                if absCosTheta(incident) > 0 {
                        fsum /= absCosTheta(incident)
                }
                return fsum
        }

        private func computeAttenuationPdf(cosThetaO: FloatX) -> [FloatX] {
                let sinThetaO = (1 - square(cosThetaO)).squareRoot()
                let sinThetaT = sinThetaO / indexRefraction
                let cosThetaT = (1 - square(sinThetaT)).squareRoot()
                let etap = (square(indexRefraction) - square(sinThetaO)).squareRoot() / cosThetaO
                let sinGammaT = hOffset / etap
                let cosGammaT = (1 - square(sinGammaT)).squareRoot()
                let transmittance = exp(-absorption * (2 * cosGammaT / cosThetaT))
                let attenuation = computeAttenutation(
                        cosThetaO: cosThetaO,
                        indexRefraction: indexRefraction,
                        hOffset: hOffset,
                        transmittance: transmittance)
                let sumY = attenuation.reduce(
                        0,
                        { s, attenuation in
                                s + attenuation.average()
                        })
                return attenuation.map { $0.average() / sumY }
        }

        func probabilityDensityLocal(outgoing: Vector, incident: Vector) -> FloatX {
                let sinThetaO = outgoing.x
                let cosThetaO = (1 - square(sinThetaO)).squareRoot()
                let phiO = atan2(outgoing.z, outgoing.y)
                let sinThetaI = incident.x
                let cosThetaI = (1 - square(sinThetaI)).squareRoot()
                let phiI = atan2(incident.z, incident.y)
                let etap = sqrt(square(indexRefraction) - square(sinThetaO)) / cosThetaO
                let sinGammaT = hOffset / etap
                let gammaT = asin(sinGammaT)
                let attenuationPdf = computeAttenuationPdf(cosThetaO: cosThetaO)

                let phi = phiI - phiO
                var pdfValue: FloatX = 0
                for pIndex in 0..<pMax {
                        let (longitudinalScattering, azimuthalScattering) = computeScattering(
                                pIndex: pIndex,
                                sinThetaI: sinThetaI,
                                cosThetaI: cosThetaI,
                                sinThetaO: sinThetaO,
                                cosThetaO: cosThetaO,
                                phi: phi,
                                gammaT: gammaT)
                        pdfValue += longitudinalScattering * attenuationPdf[pIndex] * azimuthalScattering
                }
                let longitudinalScattering = computeLongitudinalScattering(
                        cosThetaI,
                        cosThetaO,
                        sinThetaI,
                        sinThetaO,
                        vParameter[pMax])
                pdfValue += longitudinalScattering * attenuationPdf[pMax] * (1 / (2 * FloatX.pi))
                return pdfValue
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

        private func demux(_ fValue: FloatX) -> (FloatX, FloatX) {
                let bitsValue = UInt(fValue * FloatX((UInt(1) << 32)))
                let bits: (UInt32, UInt32) = (compact1by1(UInt32(bitsValue)), compact1by1(UInt32(bitsValue >> 1)))
                return (FloatX(bits.0) / FloatX(1 << 16), FloatX(bits.1) / FloatX(1 << 16))
        }

        private func sampleTrimmedLogistic(uSample: FloatX, sVal: FloatX, lower: FloatX, upper: FloatX) -> FloatX {
                let kValue = logisticCDF(upper, sVal) - logisticCDF(lower, sVal)
                let xValue = -sVal * log(1 / (uSample * kValue + logisticCDF(lower, sVal)) - 1)
                return clamp(value: xValue, low: lower, high: upper)
        }

        // func sampleLocal(wo: Vector, samples: FourRandomVariables, evaluate: (Vector, Vector)
        //  -> RgbSpectrum) async -> (
        //        RgbSpectrum, Vector, FloatX
        // ) {
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
        // }

        func albedo() -> RgbSpectrum {
                // Not correct but should be ok
                return absorption
        }
}
