struct HenyeyGreenstein: PhaseFunction {

        init(asymmetry: Real = 0) {
                self.asymmetry = asymmetry
        }

        private let inv4Pi: Real = 1.0 / (4.0 * Real.pi)

        func phase(cosTheta: Real, asymmetry: Real) -> Real {
                let denom = 1 + asymmetry * asymmetry + 2 * asymmetry * cosTheta
                return inv4Pi * (1 - asymmetry * asymmetry) / (denom * (denom).squareRoot())
        }

        func samplePhase(outgoing: Vector, sampler: inout Sampler) -> (value: Real, incident: Vector) {
                let uSample = sampler.get2D()
                var cosTheta: Real
                if abs(asymmetry) < 1e-3 {
                        cosTheta = 1 - 2 * uSample.0
                } else {
                        let sqrTerm =
                                (1 - asymmetry * asymmetry)
                                / (1 + asymmetry - 2 * asymmetry * uSample.0)
                        cosTheta =
                                -(1 + asymmetry * asymmetry - sqrTerm * sqrTerm)
                                / (2 * asymmetry)
                }
                let sinTheta = (max(0.0, 1 - cosTheta * cosTheta)).squareRoot()
                let phi = 2 * Real.pi * uSample.1
                let (vector1, vector2) = makeCoordinateSystem(from: outgoing)
                let frame = ShadingFrame(x: vector1, y: vector2, z: outgoing)
                let incident = sphericalDirection(
                        sinTheta: sinTheta,
                        cosTheta: cosTheta,
                        phi: phi,
                        frame: frame)

                let value = phase(cosTheta: cosTheta, asymmetry: asymmetry)
                return (value, incident)
        }

        func evaluate(outgoing: Vector, incident: Vector) -> Real {
                return phase(cosTheta: dot(outgoing, incident), asymmetry: asymmetry)
        }

        let asymmetry: Real
}
