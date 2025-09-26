final class HenyeyGreenstein: PhaseFunction {

        init(g: FloatX = 0) {
                self.g = g
        }

        private let inv4Pi: FloatX = 1.0 / (4.0 * FloatX.pi)

        func phase(cosTheta: FloatX, g: FloatX) -> FloatX {
                let denom = 1 + g * g + 2 * g * cosTheta
                return inv4Pi * (1 - g * g) / (denom * (denom).squareRoot())
        }

        func samplePhase(wo: Vector, sampler: RandomSampler) -> (value: FloatX, wi: Vector) {
                let u = sampler.get2D()
                var cosTheta: FloatX
                if abs(g) < 1e-3 {
                        cosTheta = 1 - 2 * u.0
                } else {
                        let sqrTerm = (1 - g * g) / (1 + g - 2 * g * u.0)
                        cosTheta = -(1 + g * g - sqrTerm * sqrTerm) / (2 * g)
                }
                let sinTheta = (max(0.0, 1 - cosTheta * cosTheta)).squareRoot()
                let phi = 2 * FloatX.pi * u.1
                let (v1, v2) = makeCoordinateSystem(from: wo)
                let wi = sphericalDirection(
                        sinTheta: sinTheta,
                        cosTheta: cosTheta,
                        phi: phi,
                        x: v1,
                        y: v2,
                        z: wo)

                let value = phase(cosTheta: cosTheta, g: g)
                return (value, wi)
        }

        func evaluate(wo: Vector, wi: Vector) -> FloatX {
                return phase(cosTheta: dot(wo, wi), g: g)
        }

        let g: FloatX
}
