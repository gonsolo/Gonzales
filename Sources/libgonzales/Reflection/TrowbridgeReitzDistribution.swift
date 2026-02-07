import Foundation  // sin, cos

public final class TrowbridgeReitzDistribution: MicrofacetDistribution {

        public init(alpha: (FloatX, FloatX)) {
                self.alpha = alpha
        }

        public func differentialArea(withNormal half: Vector) -> FloatX {
                let tan2 = tan2Theta(half)
                if tan2.isInfinite { return 0 }
                let cos4 = cos2Theta(half) * cos2Theta(half)
                let exponent = tan2 * (square(cosPhi(half) / alpha.0) + square(sinPhi(half) / alpha.1))
                let area = 1 / (FloatX.pi * alpha.0 * alpha.1 * cos4 * square(1 + exponent))
                return area
        }

        public func lambda(_ vector: Vector) -> FloatX {
                let absTanTheta = abs(tanTheta(vector))
                if absTanTheta.isInfinite { return 0 }
                let alphaTerm = (cos2Phi(vector) * alpha.0 * alpha.0 + sin2Phi(vector) * alpha.1 * alpha.1)
                        .squareRoot()
                let alpha2Tan2Theta = (alphaTerm * absTanTheta) * (alphaTerm * absTanTheta)
                let result = (-1 + (1 + alpha2Tan2Theta).squareRoot()) / 2
                return result
        }

        public func sampleHalfVector(outgoing: Vector, uSample: TwoRandomVariables) -> Vector {
                var localWo = outgoing
                let flip = outgoing.z < 0
                if flip { localWo = -localWo }
                let half = trowbridgeReitzSample(incident: localWo, alpha: alpha, uSample: uSample)
                if flip {
                        return -half
                } else {
                        return half
                }
        }

        private func trowbridgeReitzSample11(cosTheta: FloatX, uSample: TwoRandomVariables) -> (FloatX, FloatX) {
                if cosTheta > 0.9999 {
                        let radius = (uSample.0 / (1 - uSample.0)).squareRoot()
                        let phi = 2 * FloatX.pi * uSample.1
                        let slope = (radius * cos(phi), radius * sin(phi))
                        return slope
                }
                let sinTheta = (max(0, 1 - cosTheta * cosTheta)).squareRoot()
                let tanTheta = sinTheta / cosTheta
                let invTanTheta = 1 / tanTheta
                let maskingValue = 2 / (1 + (1 + 1 / (invTanTheta * invTanTheta)).squareRoot())
                let alphaTerm = 2 * uSample.0 / maskingValue - 1
                var tmp = 1 / (alphaTerm * alphaTerm - 1)
                if tmp > 1e10 { tmp = 1e10 }
                let tanThetaValue = tanTheta
                let discriminantValue = (max(0, tanThetaValue * tanThetaValue * tmp * tmp
                        - (alphaTerm * alphaTerm - tanThetaValue * tanThetaValue) * tmp)).squareRoot()
                let slopeX1 = tanThetaValue * tmp - discriminantValue
                let slopeX2 = tanThetaValue * tmp + discriminantValue
                var slope: (FloatX, FloatX) = (0.0, 0.0)
                if alphaTerm < 0 || slopeX2 > 1 / tanTheta {
                        slope.0 = slopeX1
                } else {
                        slope.0 = slopeX2
                }
                var sign: FloatX = 0.0
                var uMutable = uSample
                if uMutable.1 > 0.5 {
                        sign = 1
                        uMutable.1 = 2 * (uMutable.1 - 0.5)
                } else {
                        sign = -1
                        uMutable.1 = 2 * (0.5 - uMutable.1)
                }
                let z =
                        (uMutable.1 * (uMutable.1 * (uMutable.1 * 0.27385 - 0.73369) + 0.46341))
                        / (uMutable.1 * (uMutable.1 * (uMutable.1 * 0.093073 + 0.309420) - 1.000000) + 0.597999)
                slope.1 = sign * z * (1 + slope.0 * slope.0).squareRoot()
                assert(!slope.1.isInfinite)
                assert(!slope.1.isNaN)
                return slope
        }

        private func trowbridgeReitzSample(incident: Vector, alpha: (FloatX, FloatX), uSample: TwoRandomVariables)
                -> Vector
        {
                let incidentStretched = normalized(
                        Vector(x: alpha.0 * incident.x, y: alpha.1 * incident.y, z: incident.z))
                var slope = trowbridgeReitzSample11(cosTheta: cosTheta(incidentStretched), uSample: uSample)
                let tmp = cosPhi(incidentStretched) * slope.0 - sinPhi(incidentStretched) * slope.1
                slope.1 = sinPhi(incidentStretched) * slope.0 + cosPhi(incidentStretched) * slope.1
                slope.0 = tmp
                slope.0 = alpha.0 * slope.0
                slope.1 = alpha.1 * slope.1
                return normalized(Vector(x: -slope.0, y: -slope.1, z: 1))
        }

        static func getAlpha(from roughness: (FloatX, FloatX)) -> (FloatX, FloatX) {
                return (getAlpha(from: roughness.0), getAlpha(from: roughness.1))
        }

        static func getAlpha(from roughness: FloatX) -> FloatX {
                return roughness.squareRoot()
        }

        public var isSmooth: Bool {
                return max(alpha.0, alpha.1) < 0.001
        }

        // α = √2 σ/τ
        // σ = height
        // τ = area/length
        // α  corresponds to roughness: 0.1 is smooth (spiky highlight), 1 is Lambertian.
        let alpha: (FloatX, FloatX)
}
