final class HenyeyGreenstein: PhaseFunction {

        init(g: FloatX = 0) {
                self.g = g
        }

        private let inv4Pi: FloatX = 1.0 / (4.0 * FloatX.pi)

        func phase(cosTheta: FloatX, g: FloatX) -> FloatX {
                let denom = 1 + g * g + 2 * g * cosTheta
                return inv4Pi * (1 - g * g) / (denom * (denom).squareRoot())
        }

        func samplePhase(outgoing: Vector, sampler: inout Sampler) -> (value: FloatX, incident: Vector) {
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
                let (vector1, vector2) = makeCoordinateSystem(from: outgoing)
                let incident = sphericalDirection(
                        sinTheta: sinTheta,
                        cosTheta: cosTheta,
                        phi: phi,
                        x: vector1,
                        y: vector2,
                        z: outgoing)

                let value = phase(cosTheta: cosTheta, g: g)
                return (value, incident)
        }

        func evaluate(outgoing: Vector, incident: Vector) -> FloatX {
                return phase(cosTheta: dot(outgoing, incident), g: g)
        }

        let g: FloatX
}
