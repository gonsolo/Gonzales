///        A type that describes the reflected light from a surface based on the
///        theory that the surface consists of many differently oriented perfect
///        specular microfacets.
protocol MicrofacetDistribution {

        // D in PBRT
        func differentialArea(withNormal: Vector) -> FloatX

        // G in PBRT
        func visibleFraction(from wo: Vector, and wi: Vector) -> FloatX

        func lambda(_ vector: Vector) -> FloatX

        func sampleHalfVector(wo: Vector, u: Point2F) -> Vector

        // G1 in PBRT
        func maskingShadowing(_ vector: Vector) -> FloatX

        var isSmooth: Bool { get }

        func pdf(wo: Vector, half: Vector) -> FloatX
}

extension MicrofacetDistribution {

        func maskingShadowing(_ vector: Vector) -> FloatX {
                let result = 1 / (1 + lambda(vector))
                return result
        }

        func visibleFraction(from wo: Vector, and wi: Vector) -> FloatX {
                return 1 / (1 + lambda(wo) + lambda(wi))
        }

        func pdf(wo: Vector, half: Vector) -> FloatX {
                return differentialArea(withNormal: half) * maskingShadowing(wo) * absDot(wo, half)
                        / absCosTheta(wo)
        }
}
