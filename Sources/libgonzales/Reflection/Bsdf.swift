public enum TransportMode {
        case radiance
        case importance
}

///        A type that is a generalization of BRDFs (Bidirectional Reflection
///        Distribution Functions) and BTDFs (Bidirectional Transmission
///        Distribution Functions).
public protocol Bsdf {

        func albedo() -> RgbSpectrum
        func evaluate(outgoing: Vector, incident: Vector) -> RgbSpectrum
        func probabilityDensity(outgoing: Vector, incident: Vector) -> Real
        func sample(outgoing: Vector, uSample: ThreeRandomVariables) -> BsdfSample
        func sample(outgoing: Vector, uSample: ThreeRandomVariables, mode: TransportMode) -> BsdfSample

        var isReflective: Bool { get }
        var isTransmissive: Bool { get }
}

extension Bsdf {

        public func sample(
                outgoing: Vector,
                uSample: ThreeRandomVariables,
                evaluate: (Vector, Vector) -> RgbSpectrum
        ) -> BsdfSample {
                var incident = cosineSampleHemisphere(uSample: TwoRandomVariables(uSample.0, uSample.1))
                if outgoing.z < 0 { incident.z = -incident.z }
                let density = probabilityDensity(outgoing: outgoing, incident: incident)
                let radiance = evaluate(outgoing, incident)
                return BsdfSample(radiance, incident, density)
        }

        public func sample(outgoing: Vector, uSample: ThreeRandomVariables) -> BsdfSample {
                return sample(outgoing: outgoing, uSample: uSample, evaluate: self.evaluate)
        }

        public func sample(outgoing: Vector, uSample: ThreeRandomVariables, mode _: TransportMode)
                -> BsdfSample
        {
                return sample(outgoing: outgoing, uSample: uSample)
        }

        public func probabilityDensity(outgoing: Vector, incident: Vector) -> Real {
                guard sameHemisphere(outgoing, incident) else { return 0 }
                let result = absCosTheta(incident) / Real.pi
                return result
        }

        public var isReflective: Bool {
                return true
        }

        public var isTransmissive: Bool {
                return false
        }
}
