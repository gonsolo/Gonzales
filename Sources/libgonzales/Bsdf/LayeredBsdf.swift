import Foundation

struct LayeredBsdf<Top: Bsdf & Sendable, Bottom: Bsdf & Sendable>: FramedBsdf {

        let top: Top
        let bottom: Bottom
        let thickness: Real

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
                thickness: Real,
                albedo: RgbSpectrum,
                geometricTerm: Real,
                maxDepth: Int,
                nSamples: Int,
                bsdfFrame: BsdfFrame,
                twoSided: Bool = true
        ) {
                self.top = top
                self.bottom = bottom
                self.topIsSpecular = topIsSpecular
                self.bottomIsSpecular = bottomIsSpecular
                self.thickness = max(thickness, Real.leastNormalMagnitude)
                self._albedo = albedo
                self.phaseFunction = HenyeyGreenstein(geometricTerm: geometricTerm)
                self.maxDepth = maxDepth
                self.nSamples = nSamples
                self.bsdfFrame = bsdfFrame
                self.twoSided = twoSided
        }

        func albedo() -> RgbSpectrum {
                return self._albedo
        }

        func evaluate(outgoing: Vector, incident: Vector) -> RgbSpectrum {

                var scatteredRadiance = RgbSpectrum(intensity: 0.0)

                var localOutgoing = outgoing
                var localIncident = incident

                if twoSided && localOutgoing.z < 0 {
                        localOutgoing = -localOutgoing
                        localIncident = -localIncident
                }

                let enteredTop = twoSided || localOutgoing.z > 0
                let isSameHemisphere = sameHemisphere(localOutgoing, localIncident)
                let exitZ: Real = (isSameHemisphere != enteredTop) ? 0 : thickness

                if isSameHemisphere {
                        let entranceEval: RgbSpectrum
                        if enteredTop {
                                entranceEval = top.evaluate(outgoing: localOutgoing, incident: localIncident)
                        } else {
                                entranceEval = bottom.evaluate(outgoing: localOutgoing, incident: localIncident)
                        }
                        scatteredRadiance += Real(nSamples) * entranceEval
                }

                var sampler: Sampler = .random(RandomSampler())

                for _ in 0..<nSamples {
                        let params = SampleParams(
                                localOutgoing: localOutgoing,
                                localIncident: localIncident,
                                enteredTop: enteredTop,
                                isSameHemisphere: isSameHemisphere,
                                exitZ: exitZ)
                        scatteredRadiance += evaluateSample(
                                params: params,
                                sampler: &sampler)
                }

                let result = scatteredRadiance / Real(nSamples)
                return result
        }

        func sample(outgoing: Vector, uSample: ThreeRandomVariables) -> BsdfSample {
                var localOutgoing = outgoing
                var flipIncident = false

                if twoSided && localOutgoing.z < 0 {
                        localOutgoing = -localOutgoing
                        flipIncident = true
                }

                let enteredTop = twoSided || localOutgoing.z > 0

                let bsStart: BsdfSample
                if enteredTop {
                        bsStart = top.sample(outgoing: localOutgoing, uSample: uSample)
                } else {
                        bsStart = bottom.sample(outgoing: localOutgoing, uSample: uSample)
                }

                if !bsStart.isValid || bsStart.probabilityDensity == 0 || bsStart.incoming.z == 0 {
                        return invalidBsdfSample
                }

                if bsStart.isReflection(outgoing: localOutgoing) {
                        var resultIncident = bsStart.incoming
                        if flipIncident { resultIncident = -resultIncident }
                        return BsdfSample(bsStart.estimate, resultIncident, bsStart.probabilityDensity)
                }

                var sampledDirection = bsStart.incoming

                var sampler = Sampler.random(RandomSampler())

                var throughput = bsStart.estimate * absCosTheta(bsStart.incoming)
                var pdf = bsStart.probabilityDensity
                var zCurrent = enteredTop ? thickness : 0

                for depth in 0..<maxDepth {
                        let rrBeta = throughput.maxValue / pdf
                        if depth > 3 && rrBeta < 0.25 {
                                let rrProbability = max(0, 1 - rrBeta)
                                if sampler.get1D() < rrProbability { return invalidBsdfSample }
                                pdf *= 1 - rrProbability
                        }

                        if sampledDirection.z == 0 { return invalidBsdfSample }

                        if !_albedo.isBlack {
                                let sigmaTotal: Real = 1.0
                                let deltaZ = -log(1 - sampler.get1D()) / (sigmaTotal / absCosTheta(sampledDirection))
                                let proposedZ = sampledDirection.z > 0 ? (zCurrent + deltaZ) : (zCurrent - deltaZ)

                                if proposedZ == zCurrent { return invalidBsdfSample }

                                if proposedZ > 0 && proposedZ < thickness {
                                        let (phaseVal, nextWi) = phaseFunction.samplePhase(
                                                outgoing: -sampledDirection, sampler: &sampler)

                                        if phaseVal == 0 || nextWi.z == 0 { return invalidBsdfSample }

                                        throughput *= _albedo * phaseVal
                                        pdf *= phaseVal
                                        sampledDirection = nextWi
                                        zCurrent = proposedZ
                                        continue
                                }
                                zCurrent = clamp(value: proposedZ, low: 0, high: thickness)
                        } else {
                                zCurrent = (zCurrent == thickness) ? 0 : thickness
                                throughput *= white
                        }

                        var bsdfSample: BsdfSample
                        if zCurrent == 0 {
                                bsdfSample = bottom.sample(outgoing: -sampledDirection, uSample: sampler.get3D())
                        } else {
                                bsdfSample = top.sample(outgoing: -sampledDirection, uSample: sampler.get3D())
                        }

                        if !bsdfSample.isValid || bsdfSample.probabilityDensity == 0 || bsdfSample.incoming.z == 0 {
                                return invalidBsdfSample
                        }

                        throughput *= bsdfSample.estimate
                        pdf *= bsdfSample.probabilityDensity
                        sampledDirection = bsdfSample.incoming

                        if bsdfSample.isTransmission(outgoing: -sampledDirection) {
                                if flipIncident { sampledDirection = -sampledDirection }
                                return BsdfSample(throughput, sampledDirection, pdf)
                        }

                        throughput *= absCosTheta(bsdfSample.incoming)
                }

                return invalidBsdfSample
        }

        public func probabilityDensity(outgoing: Vector, incident: Vector) -> Real {
                var localOutgoing = outgoing
                var localIncident = incident
                if twoSided && localOutgoing.z < 0 {
                        localOutgoing = -localOutgoing
                        localIncident = -localIncident
                }

                var sampler = RandomSampler()
                let enteredTop = twoSided || localOutgoing.z > 0
                var pdfSum: Real = 0

                if sameHemisphere(localOutgoing, localIncident) {
                        let rPdf: Real
                        if enteredTop {
                                rPdf = top.probabilityDensity(outgoing: localOutgoing, incident: localIncident)
                        } else {
                                rPdf = bottom.probabilityDensity(outgoing: localOutgoing, incident: localIncident)
                        }
                        pdfSum += Real(nSamples) * rPdf
                }

                for _ in 0..<nSamples {
                        if sameHemisphere(localOutgoing, localIncident) {
                                let outgoingSample: BsdfSample
                                let incidentSample: BsdfSample
                                let rInterfacePdf: Real

                                if enteredTop {
                                        outgoingSample = top.sample(
                                                outgoing: localOutgoing, uSample: sampler.get3D())
                                        incidentSample = top.sample(
                                                outgoing: localIncident, uSample: sampler.get3D())

                                        if outgoingSample.isValid
                                                && outgoingSample.isTransmission(outgoing: localOutgoing)
                                                && incidentSample.isValid
                                                && incidentSample.isTransmission(outgoing: localIncident) {
                                                rInterfacePdf = bottom.probabilityDensity(
                                                        outgoing: -outgoingSample.incoming,
                                                        incident: -incidentSample.incoming)
                                                pdfSum += rInterfacePdf
                                        }
                                } else {
                                        outgoingSample = bottom.sample(
                                                outgoing: localOutgoing, uSample: sampler.get3D())
                                        incidentSample = bottom.sample(
                                                outgoing: localIncident, uSample: sampler.get3D())

                                        if outgoingSample.isValid
                                                && outgoingSample.isTransmission(outgoing: localOutgoing)
                                                && incidentSample.isValid
                                                && incidentSample.isTransmission(outgoing: localIncident) {
                                                rInterfacePdf = top.probabilityDensity(
                                                        outgoing: -outgoingSample.incoming,
                                                        incident: -incidentSample.incoming)
                                                pdfSum += rInterfacePdf
                                        }
                                }
                        } else {
                                if enteredTop {
                                        let outgoingSample = top.sample(
                                                outgoing: localOutgoing, uSample: sampler.get3D())
                                        let incidentSample = bottom.sample(
                                                outgoing: localIncident, uSample: sampler.get3D())

                                        if outgoingSample.isValid
                                                && !outgoingSample.isReflection(outgoing: localOutgoing)
                                                && incidentSample.isValid
                                                && !incidentSample.isReflection(outgoing: localIncident) {
                                                let probability1 = top.probabilityDensity(
                                                        outgoing: localOutgoing, incident: -incidentSample.incoming)
                                                let probability2 = bottom.probabilityDensity(
                                                        outgoing: -outgoingSample.incoming, incident: localIncident)
                                                pdfSum += (probability1 + probability2) / 2
                                        }
                                } else {
                                        let outgoingSample = bottom.sample(
                                                outgoing: localOutgoing, uSample: sampler.get3D())
                                        let incidentSample = top.sample(
                                                outgoing: localIncident, uSample: sampler.get3D())

                                        if outgoingSample.isValid
                                                && !outgoingSample.isReflection(outgoing: localOutgoing)
                                                && incidentSample.isValid
                                                && !incidentSample.isReflection(outgoing: localIncident) {
                                                let probability1 = bottom.probabilityDensity(
                                                        outgoing: localOutgoing, incident: -incidentSample.incoming)
                                                let probability2 = top.probabilityDensity(
                                                        outgoing: -outgoingSample.incoming, incident: localIncident)
                                                pdfSum += (probability1 + probability2) / 2
                                        }
                                }
                        }
                }

                return lerp(
                        with: 0.9,
                        between: 1 / (4 * Real.pi),
                        and: pdfSum / Real(nSamples))
        }

}

extension LayeredBsdf {

        private struct SampleParams {
                let localOutgoing: Vector
                let localIncident: Vector
                let enteredTop: Bool
                let isSameHemisphere: Bool
                let exitZ: Real
        }

        private func evaluateSample(
                params: SampleParams,
                sampler: inout Sampler
        ) -> RgbSpectrum {
                let localOutgoing = params.localOutgoing
                let localIncident = params.localIncident
                let enteredTop = params.enteredTop
                let isSameHemisphere = params.isSameHemisphere
                let exitZ = params.exitZ
                let uSample = sampler.get3D()

                let outgoingSample: BsdfSample
                if enteredTop {
                        outgoingSample = top.sample(outgoing: localOutgoing, uSample: uSample)
                } else {
                        outgoingSample = bottom.sample(outgoing: localOutgoing, uSample: uSample)
                }

                if !outgoingSample.isValid || outgoingSample.isReflection(outgoing: localOutgoing)
                        || outgoingSample.incoming.z == 0 {
                        return RgbSpectrum(intensity: 0)
                }

                let exitSample = sampler.get3D()
                let incidentSample: BsdfSample

                if isSameHemisphere != enteredTop {
                        incidentSample = bottom.sample(outgoing: localIncident, uSample: exitSample)
                } else {
                        incidentSample = top.sample(outgoing: localIncident, uSample: exitSample)
                }

                if !incidentSample.isValid || incidentSample.isReflection(outgoing: localIncident)
                        || incidentSample.incoming.z == 0 {
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
                exitZ: Real,
                sampler: inout Sampler
        ) -> RgbSpectrum {
                var throughput = outgoingSample.estimate * absCosTheta(outgoingSample.incoming)
                        / outgoingSample.probabilityDensity
                var zCurrent = enteredTop ? thickness : 0
                var sampledDirection = outgoingSample.incoming
                var sampleF = RgbSpectrum(intensity: 0)

                for depth in 0..<maxDepth {
                        if depth > 3 && throughput.maxValue < 0.25 {
                                let rrProbability = max(0, 1 - throughput.maxValue)
                                if sampler.get1D() < rrProbability { break }
                                throughput /= 1 - rrProbability
                        }

                        if _albedo.isBlack {
                                zCurrent = (zCurrent == thickness) ? 0 : thickness
                                throughput *= white
                        } else {
                                let sigmaTotal: Real = 1.0
                                let deltaZ = -log(1 - sampler.get1D()) / (sigmaTotal / absCosTheta(sampledDirection))
                                let proposedZ = sampledDirection.z > 0 ? (zCurrent + deltaZ) : (zCurrent - deltaZ)

                                if zCurrent == proposedZ { continue }

                                if proposedZ > 0 && proposedZ < thickness {

                                        let transmitted: Real = 1

                                        let phaseVal = phaseFunction.evaluate(
                                                outgoing: -sampledDirection, incident: -incidentSample.incoming)

                                        let transmittanceValue = transmittance(
                                                deltaZ: proposedZ - exitZ,
                                                direction: incidentSample.incoming)
                                        let term1 = throughput * _albedo * phaseVal * transmitted
                                        let term2 = transmittanceValue * incidentSample.estimate
                                                / incidentSample.probabilityDensity
                                        sampleF += term1 * term2

                                        let (phasePdf, nextWi) = phaseFunction.samplePhase(
                                                outgoing: -sampledDirection, sampler: &sampler)

                                        if phasePdf == 0 || nextWi.z == 0 { continue }

                                        throughput *= _albedo
                                        sampledDirection = nextWi
                                        zCurrent = proposedZ
                                        continue
                                }
                                zCurrent = clamp(value: proposedZ, low: 0, high: thickness)
                        }

                        if zCurrent == exitZ {
                                if let result = handleExit(
                                        zCurrent: zCurrent,
                                        sampledDirection: &sampledDirection,
                                        throughput: &throughput,
                                        sampler: &sampler) {
                                        if result { break } else { continue }
                                }
                        } else {
                                let params = NonExitParams(zCurrent: zCurrent, incidentSample: incidentSample)
                                if let result = handleNonExit(
                                        params: params,
                                        sampledDirection: &sampledDirection,
                                        throughput: &throughput,
                                        sampleF: &sampleF,
                                        sampler: &sampler) {
                                        if result { break } else { continue }
                                }
                        }
                }
                return sampleF
        }

        private func handleExit(
                zCurrent: Real,
                sampledDirection: inout Vector,
                throughput: inout RgbSpectrum,
                sampler: inout Sampler
        ) -> Bool? {
                let exitSample: BsdfSample
                if zCurrent == 0 {
                        exitSample = bottom.sample(
                                outgoing: -sampledDirection, uSample: sampler.get3D())
                } else {
                        exitSample = top.sample(
                                outgoing: -sampledDirection, uSample: sampler.get3D())
                }

                if !exitSample.isValid || exitSample.probabilityDensity == 0
                        || exitSample.incoming.z == 0 {
                        return true
                }
                if exitSample.isTransmission(outgoing: -sampledDirection) { return true }

                throughput *=
                        exitSample.estimate * absCosTheta(exitSample.incoming)
                                / exitSample.probabilityDensity
                sampledDirection = exitSample.incoming
                return nil
        }

        private struct NonExitParams {
                let zCurrent: Real
                let incidentSample: BsdfSample
        }

        private func handleNonExit(
                params: NonExitParams,
                sampledDirection: inout Vector,
                throughput: inout RgbSpectrum,
                sampleF: inout RgbSpectrum,
                sampler: inout Sampler
        ) -> Bool? {
                let zCurrent = params.zCurrent
                let incidentSample = params.incidentSample
                let isBottom = (zCurrent == 0)
                let nonExitIsSpecular = isBottom ? bottomIsSpecular : topIsSpecular

                if !nonExitIsSpecular {
                        let neVal: RgbSpectrum
                        if isBottom {
                                neVal = bottom.evaluate(
                                        outgoing: -sampledDirection, incident: -incidentSample.incoming)
                        } else {
                                neVal = top.evaluate(
                                        outgoing: -sampledDirection, incident: -incidentSample.incoming)
                        }

                        if !neVal.isBlack {
                                let transmittanceValue = transmittance(
                                        deltaZ: thickness,
                                        direction: incidentSample.incoming)
                                let term1 = throughput * neVal * absCosTheta(incidentSample.incoming)
                                let term2 = transmittanceValue * incidentSample.estimate
                                        / incidentSample.probabilityDensity
                                sampleF += term1 * term2
                        }
                }

                let sample: BsdfSample
                if isBottom {
                        sample = bottom.sample(
                                outgoing: -sampledDirection, uSample: sampler.get3D())
                } else {
                        sample = top.sample(
                                outgoing: -sampledDirection, uSample: sampler.get3D())
                }

                if !sample.isValid || sample.probabilityDensity == 0
                        || sample.incoming.z == 0 {
                        return true
                }
                if sample.isTransmission(outgoing: -sampledDirection) { return true }

                throughput *=
                        sample.estimate * absCosTheta(sample.incoming)
                                / sample.probabilityDensity
                sampledDirection = sample.incoming
                return nil
        }

        private func transmittance(deltaZ: Real, direction: Vector) -> RgbSpectrum {
                if abs(deltaZ) < Real.leastNormalMagnitude {
                        return RgbSpectrum(intensity: 1.0)
                }
                let val = abs(deltaZ / direction.z)
                let transmittanceValue = Real(exp(Float(-val)))
                return RgbSpectrum(intensity: transmittanceValue)
        }
}
