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

        func evaluateLocal(outgoing: Vector, incident: Vector) -> RgbSpectrum {

                var f = RgbSpectrum(intensity: 0.0)

                var localOutgoing = outgoing
                var localIncident = incident

                if twoSided && localOutgoing.z < 0 {
                        localOutgoing = -localOutgoing
                        localIncident = -localIncident
                }

                let enteredTop = twoSided || localOutgoing.z > 0
                let isSameHemisphere = sameHemisphere(localOutgoing, localIncident)
                let exitZ: FloatX = (isSameHemisphere != enteredTop) ? 0 : thickness

                if isSameHemisphere {
                        let entranceEval: RgbSpectrum
                        if enteredTop {
                                entranceEval = top.evaluateLocal(outgoing: localOutgoing, incident: localIncident)
                        } else {
                                entranceEval = bottom.evaluateLocal(outgoing: localOutgoing, incident: localIncident)
                        }
                        f += FloatX(nSamples) * entranceEval
                }

                var sampler: Sampler = .random(RandomSampler())

                for _ in 0..<nSamples {
                        f += evaluateSample(
                                localOutgoing: localOutgoing,
                                localIncident: localIncident,
                                enteredTop: enteredTop,
                                isSameHemisphere: isSameHemisphere,
                                exitZ: exitZ,
                                sampler: &sampler)
                }

                let result = f / FloatX(nSamples)
                return result
        }

        func sampleLocal(outgoing: Vector, u: ThreeRandomVariables) async -> BsdfSample {
                var localOutgoing = outgoing
                var flipIncident = false

                if twoSided && localOutgoing.z < 0 {
                        localOutgoing = -localOutgoing
                        flipIncident = true
                }

                let enteredTop = twoSided || localOutgoing.z > 0

                let bsStart: BsdfSample
                if enteredTop {
                        bsStart = top.sampleLocal(outgoing: localOutgoing, u: u)
                } else {
                        bsStart = bottom.sampleLocal(outgoing: localOutgoing, u: u)
                }

                if !bsStart.isValid || bsStart.probabilityDensity == 0 || bsStart.incoming.z == 0 {
                        return invalidBsdfSample
                }

                if bsStart.isReflection(outgoing: localOutgoing) {
                        var resultIncident = bsStart.incoming
                        if flipIncident { resultIncident = -resultIncident }
                        return BsdfSample(bsStart.estimate, resultIncident, bsStart.probabilityDensity)
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
                                let sigmaTotal: FloatX = 1.0
                                let deltaZ = -log(1 - sampler.get1D()) / (sigmaTotal / absCosTheta(w))
                                let proposedZ = w.z > 0 ? (z + deltaZ) : (z - deltaZ)

                                if proposedZ == z { return invalidBsdfSample }

                                if proposedZ > 0 && proposedZ < thickness {
                                        let (phaseVal, nextWi) = phaseFunction.samplePhase(
                                                outgoing: -w, sampler: &sampler)

                                        if phaseVal == 0 || nextWi.z == 0 { return invalidBsdfSample }

                                        f *= _albedo * phaseVal
                                        pdf *= phaseVal
                                        w = nextWi
                                        z = proposedZ
                                        continue
                                }
                                z = clamp(value: proposedZ, low: 0, high: thickness)
                        } else {
                                z = (z == thickness) ? 0 : thickness
                                f *= white
                        }

                        var bsdfSample: BsdfSample
                        if z == 0 {
                                bsdfSample = bottom.sampleLocal(outgoing: -w, u: sampler.get3D())
                        } else {
                                bsdfSample = top.sampleLocal(outgoing: -w, u: sampler.get3D())
                        }

                        if !bsdfSample.isValid || bsdfSample.probabilityDensity == 0 || bsdfSample.incoming.z == 0 {
                                return invalidBsdfSample
                        }

                        f *= bsdfSample.estimate
                        pdf *= bsdfSample.probabilityDensity
                        w = bsdfSample.incoming

                        if bsdfSample.isTransmission(outgoing: -w) {
                                if flipIncident { w = -w }
                                return BsdfSample(f, w, pdf)
                        }

                        f *= absCosTheta(bsdfSample.incoming)
                }

                return invalidBsdfSample
        }

        public func probabilityDensityLocal(outgoing: Vector, incident: Vector) async -> FloatX {
                var localOutgoing = outgoing
                var localIncident = incident
                if twoSided && localOutgoing.z < 0 {
                        localOutgoing = -localOutgoing
                        localIncident = -localIncident
                }

                var sampler = RandomSampler()
                let enteredTop = twoSided || localOutgoing.z > 0
                var pdfSum: FloatX = 0

                if sameHemisphere(localOutgoing, localIncident) {
                        let rPdf: FloatX
                        if enteredTop {
                                rPdf = top.probabilityDensityLocal(outgoing: localOutgoing, incident: localIncident)
                        } else {
                                rPdf = bottom.probabilityDensityLocal(outgoing: localOutgoing, incident: localIncident)
                        }
                        pdfSum += FloatX(nSamples) * rPdf
                }

                for _ in 0..<nSamples {
                        if sameHemisphere(localOutgoing, localIncident) {
                                let outgoingSample: BsdfSample
                                let incidentSample: BsdfSample
                                let rInterfacePdf: FloatX

                                if enteredTop {
                                        outgoingSample = top.sampleLocal(outgoing: localOutgoing, u: sampler.get3D())
                                        incidentSample = top.sampleLocal(outgoing: localIncident, u: sampler.get3D())

                                        if outgoingSample.isValid && outgoingSample.isTransmission(outgoing: localOutgoing) && incidentSample.isValid
                                                && incidentSample.isTransmission(outgoing: localIncident)
                                        {
                                                rInterfacePdf = bottom.probabilityDensityLocal(
                                                        outgoing: -outgoingSample.incoming, incident: -incidentSample.incoming)
                                                pdfSum += rInterfacePdf
                                        }
                                } else {
                                        outgoingSample = bottom.sampleLocal(outgoing: localOutgoing, u: sampler.get3D())
                                        incidentSample = bottom.sampleLocal(outgoing: localIncident, u: sampler.get3D())

                                        if outgoingSample.isValid && outgoingSample.isTransmission(outgoing: localOutgoing) && incidentSample.isValid
                                                && incidentSample.isTransmission(outgoing: localIncident)
                                        {
                                                rInterfacePdf = top.probabilityDensityLocal(
                                                        outgoing: -outgoingSample.incoming, incident: -incidentSample.incoming)
                                                pdfSum += rInterfacePdf
                                        }
                                }
                        } else {
                                if enteredTop {
                                        let outgoingSample = top.sampleLocal(outgoing: localOutgoing, u: sampler.get3D())
                                        let incidentSample = bottom.sampleLocal(outgoing: localIncident, u: sampler.get3D())

                                        if outgoingSample.isValid && !outgoingSample.isReflection(outgoing: localOutgoing) && incidentSample.isValid
                                                && !incidentSample.isReflection(outgoing: localIncident)
                                        {
                                                let probability1 = top.probabilityDensityLocal(
                                                        outgoing: localOutgoing, incident: -incidentSample.incoming)
                                                let probability2 = bottom.probabilityDensityLocal(
                                                        outgoing: -outgoingSample.incoming, incident: localIncident)
                                                pdfSum += (probability1 + probability2) / 2
                                        }
                                } else {
                                        let outgoingSample = bottom.sampleLocal(outgoing: localOutgoing, u: sampler.get3D())
                                        let incidentSample = top.sampleLocal(outgoing: localIncident, u: sampler.get3D())

                                        if outgoingSample.isValid && !outgoingSample.isReflection(outgoing: localOutgoing) && incidentSample.isValid
                                                && !incidentSample.isReflection(outgoing: localIncident)
                                        {
                                                let probability1 = bottom.probabilityDensityLocal(
                                                        outgoing: localOutgoing, incident: -incidentSample.incoming)
                                                let probability2 = top.probabilityDensityLocal(
                                                        outgoing: -outgoingSample.incoming, incident: localIncident)
                                                pdfSum += (probability1 + probability2) / 2
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
                localOutgoing: Vector,
                localIncident: Vector,
                enteredTop: Bool,
                isSameHemisphere: Bool,
                exitZ: FloatX,
                sampler: inout Sampler
        ) -> RgbSpectrum {
                let u = sampler.get3D()

                let outgoingSample: BsdfSample
                if enteredTop {
                        outgoingSample = top.sampleLocal(outgoing: localOutgoing, u: u)
                } else {
                        outgoingSample = bottom.sampleLocal(outgoing: localOutgoing, u: u)
                }

                if !outgoingSample.isValid || outgoingSample.isReflection(outgoing: localOutgoing) || outgoingSample.incoming.z == 0 {
                        return RgbSpectrum(intensity: 0)
                }

                let exitSample = sampler.get3D()
                let incidentSample: BsdfSample

                if isSameHemisphere != enteredTop {
                        incidentSample = bottom.sampleLocal(outgoing: localIncident, u: exitSample)
                } else {
                        incidentSample = top.sampleLocal(outgoing: localIncident, u: exitSample)
                }

                if !incidentSample.isValid || incidentSample.isReflection(outgoing: localIncident) || incidentSample.incoming.z == 0 {
                        return RgbSpectrum(intensity: 0)
                }

                return tracePath(
                        outgoingSample: outgoingSample,
                        incidentSample: incidentSample,
                        enteredTop: enteredTop,
                        exitZ: exitZ,
                        sampler: &sampler)
        }

        private func tracePath(
                outgoingSample: BsdfSample,
                incidentSample: BsdfSample,
                enteredTop: Bool,
                exitZ: FloatX,
                sampler: inout Sampler
        ) -> RgbSpectrum {
                var beta = outgoingSample.estimate * absCosTheta(outgoingSample.incoming) / outgoingSample.probabilityDensity
                var z = enteredTop ? thickness : 0
                var w = outgoingSample.incoming
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
                                let sigmaTotal: FloatX = 1.0
                                let deltaZ = -log(1 - sampler.get1D()) / (sigmaTotal / absCosTheta(w))
                                let proposedZ = w.z > 0 ? (z + deltaZ) : (z - deltaZ)

                                if z == proposedZ { continue }

                                if proposedZ > 0 && proposedZ < thickness {

                                        let transmitted: FloatX = 1

                                        let phaseVal = phaseFunction.evaluate(
                                                outgoing: -w, incident: -incidentSample.incoming)

                                        let transmittanceValue = transmittance(
                                                deltaZ: proposedZ - exitZ,
                                                w: incidentSample.incoming)
                                        let term1 = beta * _albedo * phaseVal * transmitted
                                        let term2 = transmittanceValue * incidentSample.estimate / incidentSample.probabilityDensity
                                        sampleF += term1 * term2

                                        let (phasePdf, nextWi) = phaseFunction.samplePhase(
                                                outgoing: -w, sampler: &sampler)

                                        if phasePdf == 0 || nextWi.z == 0 { continue }

                                        beta *= _albedo
                                        w = nextWi
                                        z = proposedZ
                                        continue
                                }
                                z = clamp(value: proposedZ, low: 0, high: thickness)
                        }

                        if z == exitZ {
                                let exitSample: BsdfSample
                                if z == 0 {
                                        exitSample = bottom.sampleLocal(outgoing: -w, u: sampler.get3D())
                                } else {
                                        exitSample = top.sampleLocal(outgoing: -w, u: sampler.get3D())
                                }

                                if !exitSample.isValid || exitSample.probabilityDensity == 0
                                        || exitSample.incoming.z == 0
                                {
                                        break
                                }
                                if exitSample.isTransmission(outgoing: -w) { break }

                                beta *=
                                        exitSample.estimate * absCosTheta(exitSample.incoming)
                                        / exitSample.probabilityDensity
                                w = exitSample.incoming

                        } else {
                                let isBottom = (z == 0)
                                let nonExitIsSpecular = isBottom ? bottomIsSpecular : topIsSpecular

                                if !nonExitIsSpecular {
                                        let neVal: RgbSpectrum
                                        if isBottom {
                                                neVal = bottom.evaluateLocal(
                                                        outgoing: -w, incident: -incidentSample.incoming)
                                        } else {
                                                neVal = top.evaluateLocal(outgoing: -w, incident: -incidentSample.incoming)
                                        }

                                        if !neVal.isBlack {
                                                let transmittanceValue = transmittance(
                                                        deltaZ: thickness,
                                                        w: incidentSample.incoming)
                                                let term1 = beta * neVal * absCosTheta(incidentSample.incoming)
                                                let term2 = transmittanceValue * incidentSample.estimate / incidentSample.probabilityDensity
                                                sampleF += term1 * term2
                                        }
                                }

                                let sample: BsdfSample
                                if isBottom {
                                        sample = bottom.sampleLocal(outgoing: -w, u: sampler.get3D())
                                } else {
                                        sample = top.sampleLocal(outgoing: -w, u: sampler.get3D())
                                }

                                if !sample.isValid || sample.probabilityDensity == 0
                                        || sample.incoming.z == 0
                                {
                                        break
                                }
                                if sample.isTransmission(outgoing: -w) { break }

                                beta *=
                                        sample.estimate * absCosTheta(sample.incoming)
                                        / sample.probabilityDensity
                                w = sample.incoming
                        }
                }
                return sampleF
        }

        private func transmittance(deltaZ: FloatX, w: Vector) -> RgbSpectrum {
                if abs(deltaZ) < FloatX.leastNormalMagnitude {
                        return RgbSpectrum(intensity: 1.0)
                }
                let val = abs(deltaZ / w.z)
                let transmittanceValue = FloatX(exp(Float(-val)))
                return RgbSpectrum(intensity: transmittanceValue)
        }
}
