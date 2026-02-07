final class HenyeyGreenstein: PhaseFunction {

        init(geometricTerm: FloatX = 0) {
                self.geometricTerm = geometricTerm
        }

        private let inv4Pi: FloatX = 1.0 / (4.0 * FloatX.pi)

        func phase(cosTheta: FloatX, geometricTerm: FloatX) -> FloatX {
                let denom = 1 + geometricTerm * geometricTerm + 2 * geometricTerm * cosTheta
                return inv4Pi * (1 - geometricTerm * geometricTerm) / (denom * (denom).squareRoot())
        }

        func samplePhase(outgoing: Vector, sampler: inout Sampler) -> (value: FloatX, incident: Vector) {
                let uSample = sampler.get2D()
                var cosTheta: FloatX
                if abs(geometricTerm) < 1e-3 {
                        cosTheta = 1 - 2 * uSample.0
                } else {
                        let sqrTerm = (1 - geometricTerm * geometricTerm)
                                / (1 + geometricTerm - 2 * geometricTerm * uSample.0)
                        cosTheta = -(1 + geometricTerm * geometricTerm - sqrTerm * sqrTerm)
                                / (2 * geometricTerm)
                }
                let sinTheta = (max(0.0, 1 - cosTheta * cosTheta)).squareRoot()
                let phi = 2 * FloatX.pi * uSample.1
                let (vector1, vector2) = makeCoordinateSystem(from: outgoing)
                let frame = ShadingFrame(x: vector1, y: vector2, z: outgoing)
                let incident = sphericalDirection(
                        sinTheta: sinTheta,
                        cosTheta: cosTheta,
                        phi: phi,
                        frame: frame)

                let value = phase(cosTheta: cosTheta, geometricTerm: geometricTerm)
                return (value, incident)
        }

        func evaluate(outgoing: Vector, incident: Vector) -> FloatX {
                return phase(cosTheta: dot(outgoing, incident), geometricTerm: geometricTerm)
        }

        let geometricTerm: FloatX
}
