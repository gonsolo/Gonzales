///        A type that is a generalization of BRDFs (Bidirectional Reflection
///        Distribution Functions) and BTDFs (Bidirectional Transmission
///        Distribution Functions).
public protocol LocalBsdf {

        func albedo() -> RgbSpectrum
        func evaluateLocal(wo: Vector, wi: Vector) -> RgbSpectrum
        func probabilityDensityLocal(wo: Vector, wi: Vector) -> FloatX
        func sampleLocal(wo: Vector, u: ThreeRandomVariables) -> BsdfSample

        var isReflective: Bool { get }
        var isTransmissive: Bool { get }
}

public extension LocalBsdf {

        func sampleLocal(
                wo: Vector,
                u: ThreeRandomVariables,
                evaluate: (Vector, Vector) -> RgbSpectrum
        ) -> BsdfSample {
                var wi = cosineSampleHemisphere(u: TwoRandomVariables(u.0, u.1))
                if wo.z < 0 { wi.z = -wi.z }
                let density = probabilityDensityLocal(wo: wo, wi: wi)
                let radiance = evaluate(wo, wi)
                return BsdfSample(radiance, wi, density)
        }

        func sampleLocal(wo: Vector, u: ThreeRandomVariables) -> BsdfSample {
                return sampleLocal(wo: wo, u: u, evaluate: self.evaluateLocal)
        }

        func probabilityDensityLocal(wo: Vector, wi: Vector) -> FloatX {
                guard sameHemisphere(wo, wi) else { return 0 }
                let result = absCosTheta(wi) / FloatX.pi
                return result
        }

        var isReflective: Bool {
                return true
        }

        var isTransmissive: Bool {
                return false
        }
}
