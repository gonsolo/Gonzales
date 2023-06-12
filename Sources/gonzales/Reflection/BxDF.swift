///        A type that is a generalization of BRDFs (Bidirectional Reflection
///        Distribution Functions) and BTDFs (Bidirectional Transmission
///        Distribution Functions).
protocol LocalBsdf {

        func albedoLocal() -> RGBSpectrum
        func evaluateLocal(wo: Vector, wi: Vector) -> RGBSpectrum
        func probabilityDensityLocal(wo: Vector, wi: Vector) -> FloatX
        func sampleLocal(wo: Vector, u: ThreeRandomVariables) -> BSDFSample

        var isReflective: Bool { get }
        var isTransmissive: Bool { get }
}

extension LocalBsdf {

        func sampleLocal(
                wo: Vector,
                u: ThreeRandomVariables,
                evaluate: (Vector, Vector) -> RGBSpectrum
        ) -> BSDFSample {
                var wi = cosineSampleHemisphere(u: TwoRandomVariables(u.0, u.1))
                if wo.z < 0 { wi.z = -wi.z }
                let density = probabilityDensityLocal(wo: wo, wi: wi)
                let radiance = evaluate(wo, wi)
                return BSDFSample(radiance, wi, density)
        }

        func sampleLocal(wo: Vector, u: ThreeRandomVariables) -> BSDFSample {
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
