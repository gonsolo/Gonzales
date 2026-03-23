///        A type that describes the reflected light from a surface based on the
///        theory that the surface consists of many differently oriented perfect
///        specular microfacets.
public protocol MicrofacetDistribution: Sendable {

        // D in PBRT
        func differentialArea(withNormal: Vector) -> Real

        // G in PBRT
        func visibleFraction(from outgoing: Vector, and incident: Vector) -> Real

        func lambda(_ vector: Vector) -> Real

        func sampleHalfVector(outgoing: Vector, uSample: TwoRandomVariables) -> Vector

        // G1 in PBRT
        func maskingShadowing(_ vector: Vector) -> Real

        var isSmooth: Bool { get }

        func probabilityDensity(outgoing: Vector, half: Vector) -> Real
}

extension MicrofacetDistribution {

        public func maskingShadowing(_ vector: Vector) -> Real {
                let result = 1 / (1 + lambda(vector))
                return result
        }

        public func visibleFraction(from outgoing: Vector, and incident: Vector) -> Real {
                return 1 / (1 + lambda(outgoing) + lambda(incident))
        }

        public func probabilityDensity(outgoing: Vector, half: Vector) -> Real {
                return differentialArea(withNormal: half) * maskingShadowing(outgoing)
                        * absDot(outgoing, half)
                        / absCosTheta(outgoing)
        }
}
