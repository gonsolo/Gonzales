///        A type that is a generalization of BRDFs (Bidirectional Reflection
///        Distribution Functions) and BTDFs (Bidirectional Transmission
///        Distribution Functions).
public protocol LocalBsdf {

        func albedo() -> RgbSpectrum
        func evaluateLocal(outgoing: Vector, incident: Vector) -> RgbSpectrum
        func probabilityDensityLocal(outgoing: Vector, incident: Vector) -> FloatX
        func sampleLocal(outgoing: Vector, uSample: ThreeRandomVariables) -> BsdfSample

        var isReflective: Bool { get }
        var isTransmissive: Bool { get }
}

extension LocalBsdf {

        public func sampleLocal(
                outgoing: Vector,
                uSample: ThreeRandomVariables,
                evaluate: (Vector, Vector) -> RgbSpectrum
        ) -> BsdfSample {
                var incident = cosineSampleHemisphere(uSample: TwoRandomVariables(uSample.0, uSample.1))
                if outgoing.z < 0 { incident.z = -incident.z }
                let density = probabilityDensityLocal(outgoing: outgoing, incident: incident)
                let radiance = evaluate(outgoing, incident)
                return BsdfSample(radiance, incident, density)
        }

        public func sampleLocal(outgoing: Vector, uSample: ThreeRandomVariables) -> BsdfSample {
                return sampleLocal(outgoing: outgoing, uSample: uSample, evaluate: self.evaluateLocal)
        }

        public func probabilityDensityLocal(outgoing: Vector, incident: Vector) -> FloatX {
                guard sameHemisphere(outgoing, incident) else { return 0 }
                let result = absCosTheta(incident) / FloatX.pi
                return result
        }

        public var isReflective: Bool {
                return true
        }

        public var isTransmissive: Bool {
                return false
        }
}
