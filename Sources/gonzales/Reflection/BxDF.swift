///        A type that is a generalization of BRDFs (Bidirectional Reflection
///        Distribution Functions) and BTDFs (Bidirectional Transmission
///        Distribution Functions).
protocol BxDF: AnyObject {
        func evaluate(wo: Vector, wi: Vector) -> Spectrum
        func sample(wo: Vector, u: Point2F) -> (Spectrum, Vector, FloatX)
        func probabilityDensity(wo: Vector, wi: Vector) -> FloatX
        func albedo() -> Spectrum

        var isReflective: Bool { get }
        var isTransmissive: Bool { get }
}

extension BxDF {

        func sample(wo: Vector, u: Point2F, evaluate: (Vector, Vector) -> Spectrum) -> (
                Spectrum, Vector, FloatX
        ) {
                var wi = cosineSampleHemisphere(u: u)
                if wo.z < 0 { wi.z = -wi.z }
                let density = probabilityDensity(wo: wo, wi: wi)
                let radiance = evaluate(wo, wi)
                return (radiance, wi, density)
        }

        func sample(wo: Vector, u: Point2F) -> (Spectrum, Vector, FloatX) {
                return sample(wo: wo, u: u, evaluate: self.evaluate)
        }

        func probabilityDensity(wo: Vector, wi: Vector) -> FloatX {
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
