import Foundation

struct LayeredBsdf<Top: LocalBsdf & Sendable, Bottom: LocalBsdf & Sendable>: GlobalBsdf {

        let top: Top
        let bottom: Bottom
        let thickness: FloatX

        private let _albedo: RgbSpectrum

        let maxDepth: Int
        let nSamples: Int
        let twoSided: Bool

        let bsdfFrame: BsdfFrame

        let phaseFunction: HenyeyGreenstein

        private let topIsSpecular: Bool
        private let bottomIsSpecular: Bool
}

extension LayeredBsdf {

        init(
                top: Top,
                bottom: Bottom,
                topIsSpecular: Bool,
                bottomIsSpecular: Bool,
                thickness: FloatX,
                albedo: RgbSpectrum,
                g: FloatX,
                maxDepth: Int,
                nSamples: Int,
                bsdfFrame: BsdfFrame,
                twoSided: Bool = true
        ) {
                self.top = top
                self.bottom = bottom
                self.topIsSpecular = topIsSpecular
                self.bottomIsSpecular = bottomIsSpecular
                self.thickness = max(thickness, FloatX.leastNormalMagnitude)
                self._albedo = albedo
                self.phaseFunction = HenyeyGreenstein(g: g)
                self.maxDepth = maxDepth
                self.nSamples = nSamples
                self.bsdfFrame = bsdfFrame
                self.twoSided = twoSided
        }

        func albedo() -> RgbSpectrum {
                return self._albedo
        }

        func evaluateLocal(wo: Vector, wi: Vector) -> RgbSpectrum {

                var f = RgbSpectrum(intensity: 0.0)

                var localWo = wo
                var localWi = wi

                if twoSided && localWo.z < 0 {
                        localWo = -localWo
                        localWi = -localWi
                }

                let enteredTop = twoSided || localWo.z > 0
                let isSameHemisphere = sameHemisphere(localWo, localWi)
                let exitZ: FloatX = (isSameHemisphere != enteredTop) ? 0 : thickness

                if isSameHemisphere {
                        let entranceEval: RgbSpectrum
                        if enteredTop {
                                entranceEval = top.evaluateLocal(wo: localWo, wi: localWi)
                        } else {
                                entranceEval = bottom.evaluateLocal(wo: localWo, wi: localWi)
                        }
                        f += FloatX(nSamples) * entranceEval
                }

                var sampler: Sampler = .random(RandomSampler())

                for _ in 0..<nSamples {
                        f += evaluateSample(
                                localWo: localWo,
                                localWi: localWi,
                                enteredTop: enteredTop,
                                isSameHemisphere: isSameHemisphere,
                                exitZ: exitZ,
                                sampler: &sampler)
                }

                let result = f / FloatX(nSamples)
                return result
        }

        func sampleLocal(wo: Vector, u: ThreeRandomVariables) async -> BsdfSample {
                var localWo = wo
                var flipWi = false

                if twoSided && localWo.z < 0 {
                        localWo = -localWo
                        flipWi = true
                }

                let enteredTop = twoSided || localWo.z > 0

                let bsStart: BsdfSample
                if enteredTop {
                        bsStart = top.sampleLocal(wo: localWo, u: u)
                } else {
                        bsStart = bottom.sampleLocal(wo: localWo, u: u)
                }

                if !bsStart.isValid || bsStart.probabilityDensity == 0 || bsStart.incoming.z == 0 {
                        return invalidBsdfSample
                }

                if bsStart.isReflection(wo: localWo) {
                        var resultWi = bsStart.incoming
                        if flipWi { resultWi = -resultWi }
                        return BsdfSample(bsStart.estimate, resultWi, bsStart.probabilityDensity)
                }

                var w = bsStart.incoming

                var sampler = Sampler.random(RandomSampler())

                var f = bsStart.estimate * absCosTheta(bsStart.incoming)
                var pdf = bsStart.probabilityDensity
                var z = enteredTop ? thickness : 0

                for depth in 0..<maxDepth {
                        let rrBeta = f.maxValue / pdf
                        if depth > 3 && rrBeta < 0.25 {
                                let q = max(0, 1 - rrBeta)
                                if sampler.get1D() < q { return invalidBsdfSample }
                                pdf *= 1 - q
                        }

                        if w.z == 0 { return invalidBsdfSample }

                        if !_albedo.isBlack {
                                let sigma_t: FloatX = 1.0
                                let dz = -log(1 - sampler.get1D()) / (sigma_t / absCosTheta(w))
                                let zp = w.z > 0 ? (z + dz) : (z - dz)

                                if zp == z { return invalidBsdfSample }

                                if zp > 0 && zp < thickness {
                                        let (phaseVal, nextWi) = phaseFunction.samplePhase(
                                                wo: -w, sampler: &sampler)

                                        if phaseVal == 0 || nextWi.z == 0 { return invalidBsdfSample }

                                        f *= _albedo * phaseVal
                                        pdf *= phaseVal
                                        w = nextWi
                                        z = zp
                                        continue
                                }
                                z = clamp(value: zp, low: 0, high: thickness)
                        } else {
                                z = (z == thickness) ? 0 : thickness
                                f *= white
                        }

                        let bs: BsdfSample
                        if z == 0 {
                                bs = bottom.sampleLocal(wo: -w, u: sampler.get3D())
                        } else {
                                bs = top.sampleLocal(wo: -w, u: sampler.get3D())
                        }

                        if !bs.isValid || bs.probabilityDensity == 0 || bs.incoming.z == 0 {
                                return invalidBsdfSample
                        }

                        f *= bs.estimate
                        pdf *= bs.probabilityDensity
                        w = bs.incoming

                        if bs.isTransmission(wo: -w) {
                                if flipWi { w = -w }
                                return BsdfSample(f, w, pdf)
                        }

                        f *= absCosTheta(bs.incoming)
                }

                return invalidBsdfSample
        }

        public func probabilityDensityLocal(wo: Vector, wi: Vector) async -> FloatX {
                var localWo = wo
                var localWi = wi
                if twoSided && localWo.z < 0 {
                        localWo = -localWo
                        localWi = -localWi
                }

                var sampler = RandomSampler()
                let enteredTop = twoSided || localWo.z > 0
                var pdfSum: FloatX = 0

                if sameHemisphere(localWo, localWi) {
                        let rPdf: FloatX
                        if enteredTop {
                                rPdf = top.probabilityDensityLocal(wo: localWo, wi: localWi)
                        } else {
                                rPdf = bottom.probabilityDensityLocal(wo: localWo, wi: localWi)
                        }
                        pdfSum += FloatX(nSamples) * rPdf
                }

                for _ in 0..<nSamples {
                        if sameHemisphere(localWo, localWi) {
                                let wos: BsdfSample
                                let wis: BsdfSample
                                let rInterfacePdf: FloatX

                                if enteredTop {
                                        wos = top.sampleLocal(wo: localWo, u: sampler.get3D())
                                        wis = top.sampleLocal(wo: localWi, u: sampler.get3D())

                                        if wos.isValid && wos.isTransmission(wo: localWo) && wis.isValid
                                                && wis.isTransmission(wo: localWi)
                                        {
                                                rInterfacePdf = bottom.probabilityDensityLocal(
                                                        wo: -wos.incoming, wi: -wis.incoming)
                                                pdfSum += rInterfacePdf
                                        }
                                } else {
                                        wos = bottom.sampleLocal(wo: localWo, u: sampler.get3D())
                                        wis = bottom.sampleLocal(wo: localWi, u: sampler.get3D())

                                        if wos.isValid && wos.isTransmission(wo: localWo) && wis.isValid
                                                && wis.isTransmission(wo: localWi)
                                        {
                                                rInterfacePdf = top.probabilityDensityLocal(
                                                        wo: -wos.incoming, wi: -wis.incoming)
                                                pdfSum += rInterfacePdf
                                        }
                                }
                        } else {
                                if enteredTop {
                                        let wos = top.sampleLocal(wo: localWo, u: sampler.get3D())
                                        let wis = bottom.sampleLocal(wo: localWi, u: sampler.get3D())

                                        if wos.isValid && !wos.isReflection(wo: localWo) && wis.isValid
                                                && !wis.isReflection(wo: localWi)
                                        {
                                                let p1 = top.probabilityDensityLocal(
                                                        wo: localWo, wi: -wis.incoming)
                                                let p2 = bottom.probabilityDensityLocal(
                                                        wo: -wos.incoming, wi: localWi)
                                                pdfSum += (p1 + p2) / 2
                                        }
                                } else {
                                        let wos = bottom.sampleLocal(wo: localWo, u: sampler.get3D())
                                        let wis = top.sampleLocal(wo: localWi, u: sampler.get3D())

                                        if wos.isValid && !wos.isReflection(wo: localWo) && wis.isValid
                                                && !wis.isReflection(wo: localWi)
                                        {
                                                let p1 = bottom.probabilityDensityLocal(
                                                        wo: localWo, wi: -wis.incoming)
                                                let p2 = top.probabilityDensityLocal(
                                                        wo: -wos.incoming, wi: localWi)
                                                pdfSum += (p1 + p2) / 2
                                        }
                                }
                        }
                }

                return lerp(
                        with: 0.9,
                        between: 1 / (4 * FloatX.pi),
                        and: pdfSum / FloatX(nSamples))
        }

}

extension LayeredBsdf {

        private func evaluateSample(
                localWo: Vector,
                localWi: Vector,
                enteredTop: Bool,
                isSameHemisphere: Bool,
                exitZ: FloatX,
                sampler: inout Sampler
        ) -> RgbSpectrum {
                let u = sampler.get3D()

                let wos: BsdfSample
                if enteredTop {
                        wos = top.sampleLocal(wo: localWo, u: u)
                } else {
                        wos = bottom.sampleLocal(wo: localWo, u: u)
                }

                if !wos.isValid || wos.isReflection(wo: localWo) || wos.incoming.z == 0 {
                        return RgbSpectrum(intensity: 0)
                }

                let u_exit = sampler.get3D()
                let wis: BsdfSample

                if isSameHemisphere != enteredTop {
                        wis = bottom.sampleLocal(wo: localWi, u: u_exit)
                } else {
                        wis = top.sampleLocal(wo: localWi, u: u_exit)
                }

                if !wis.isValid || wis.isReflection(wo: localWi) || wis.incoming.z == 0 {
                        return RgbSpectrum(intensity: 0)
                }

                return tracePath(
                        wos: wos,
                        wis: wis,
                        enteredTop: enteredTop,
                        exitZ: exitZ,
                        sampler: &sampler)
        }

        private func tracePath(
                wos: BsdfSample,
                wis: BsdfSample,
                enteredTop: Bool,
                exitZ: FloatX,
                sampler: inout Sampler
        ) -> RgbSpectrum {
                var beta = wos.estimate * absCosTheta(wos.incoming) / wos.probabilityDensity
                var z = enteredTop ? thickness : 0
                var w = wos.incoming
                var sampleF = RgbSpectrum(intensity: 0)

                for depth in 0..<maxDepth {
                        if depth > 3 && beta.maxValue < 0.25 {
                                let q = max(0, 1 - beta.maxValue)
                                if sampler.get1D() < q { break }
                                beta /= 1 - q
                        }

                        if _albedo.isBlack {
                                z = (z == thickness) ? 0 : thickness
                                beta *= white
                        } else {
                                let sigma_t: FloatX = 1.0
                                let dz = -log(1 - sampler.get1D()) / (sigma_t / abs(w.z))
                                let zp = w.z > 0 ? (z + dz) : (z - dz)

                                if z == zp { continue }

                                if zp > 0 && zp < thickness {

                                        let wt: FloatX = 1

                                        let phaseVal = phaseFunction.evaluate(
                                                wo: -w, wi: -wis.incoming)

                                        let tr = transmittance(dz: zp - exitZ, w: wis.incoming)
                                        let term1 = beta * _albedo * phaseVal * wt
                                        let term2 = tr * wis.estimate / wis.probabilityDensity
                                        sampleF += term1 * term2

                                        let (phasePdf, nextWi) = phaseFunction.samplePhase(
                                                wo: -w, sampler: &sampler)

                                        if phasePdf == 0 || nextWi.z == 0 { continue }

                                        beta *= _albedo
                                        w = nextWi
                                        z = zp
                                        continue
                                }
                                z = clamp(value: zp, low: 0, high: thickness)
                        }

                        if z == exitZ {
                                let bsExit: BsdfSample
                                if z == 0 {
                                        bsExit = bottom.sampleLocal(wo: -w, u: sampler.get3D())
                                } else {
                                        bsExit = top.sampleLocal(wo: -w, u: sampler.get3D())
                                }

                                if !bsExit.isValid || bsExit.probabilityDensity == 0
                                        || bsExit.incoming.z == 0
                                {
                                        break
                                }
                                if bsExit.isTransmission(wo: -w) { break }

                                beta *=
                                        bsExit.estimate * absCosTheta(bsExit.incoming)
                                        / bsExit.probabilityDensity
                                w = bsExit.incoming

                        } else {
                                let isBottom = (z == 0)
                                let nonExitIsSpecular = isBottom ? bottomIsSpecular : topIsSpecular

                                if !nonExitIsSpecular {
                                        let neVal: RgbSpectrum
                                        if isBottom {
                                                neVal = bottom.evaluateLocal(
                                                        wo: -w, wi: -wis.incoming)
                                        } else {
                                                neVal = top.evaluateLocal(wo: -w, wi: -wis.incoming)
                                        }

                                        if !neVal.isBlack {
                                                let tr = transmittance(dz: thickness, w: wis.incoming)
                                                let term1 = beta * neVal * absCosTheta(wis.incoming)
                                                let term2 = tr * wis.estimate / wis.probabilityDensity
                                                sampleF += term1 * term2
                                        }
                                }

                                let bs: BsdfSample
                                if isBottom {
                                        bs = bottom.sampleLocal(wo: -w, u: sampler.get3D())
                                } else {
                                        bs = top.sampleLocal(wo: -w, u: sampler.get3D())
                                }

                                if !bs.isValid || bs.probabilityDensity == 0 || bs.incoming.z == 0 {
                                        break
                                }
                                if bs.isTransmission(wo: -w) { break }

                                beta *= bs.estimate * absCosTheta(bs.incoming) / bs.probabilityDensity
                                w = bs.incoming
                        }
                }
                return sampleF
        }

        private func transmittance(dz: FloatX, w: Vector) -> RgbSpectrum {
                if abs(dz) < FloatX.leastNormalMagnitude {
                        return RgbSpectrum(intensity: 1.0)
                }
                let val = abs(dz / w.z)
                let tr = FloatX(exp(Float(-val)))
                return RgbSpectrum(intensity: tr)
        }
}
