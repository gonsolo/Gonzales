import Foundation  // sin, cos

public final class TrowbridgeReitzDistribution: MicrofacetDistribution {

        public init(alpha: (FloatX, FloatX)) {
                self.alpha = alpha
        }

        public func differentialArea(withNormal half: Vector) -> FloatX {
                let tan2 = tan2Theta(half)
                if tan2.isInfinite { return 0 }
                let cos4 = cos2Theta(half) * cos2Theta(half)
                let e = tan2 * (square(cosPhi(half) / alpha.0) + square(sinPhi(half) / alpha.1))
                let area = 1 / (FloatX.pi * alpha.0 * alpha.1 * cos4 * square(1 + e))
                return area
        }

        public func lambda(_ vector: Vector) -> FloatX {
                let absTanTheta = abs(tanTheta(vector))
                if absTanTheta.isInfinite { return 0 }
                let a = (cos2Phi(vector) * alpha.0 * alpha.0 + sin2Phi(vector) * alpha.1 * alpha.1)
                        .squareRoot()
                let alpha2Tan2Theta = (a * absTanTheta) * (a * absTanTheta)
                let result = (-1 + (1 + alpha2Tan2Theta).squareRoot()) / 2
                return result
        }

        public func sampleHalfVector(wo: Vector, u: TwoRandomVariables) -> Vector {
                var localWo = wo
                let flip = wo.z < 0
                if flip { localWo = -localWo }
                let half = trowbridgeReitzSample(wi: localWo, alpha: alpha, u: u)
                if flip {
                        return -half
                } else {
                        return half
                }
        }

        private func trowbridgeReitzSample11(cosTheta: FloatX, u: TwoRandomVariables) -> (FloatX, FloatX) {
                if cosTheta > 0.9999 {
                        let r = (u.0 / (1 - u.0)).squareRoot()
                        let phi = 2 * FloatX.pi * u.1
                        let slope = (r * cos(phi), r * sin(phi))
                        return slope
                }
                let sinTheta = (max(0, 1 - cosTheta * cosTheta)).squareRoot()
                let tanTheta = sinTheta / cosTheta
                let invTanTheta = 1 / tanTheta
                let g1 = 2 / (1 + (1 + 1 / (invTanTheta * invTanTheta)).squareRoot())
                let a = 2 * u.0 / g1 - 1
                var tmp = 1 / (a * a - 1)
                if tmp > 1e10 { tmp = 1e10 }
                let b = tanTheta
                let d = (max(0, b * b * tmp * tmp - (a * a - b * b) * tmp)).squareRoot()
                let slopeX1 = b * tmp - d
                let slopeX2 = b * tmp + d
                var slope: (FloatX, FloatX) = (0.0, 0.0)
                if a < 0 || slopeX2 > 1 / tanTheta {
                        slope.0 = slopeX1
                } else {
                        slope.0 = slopeX2
                }
                var s: FloatX = 0.0
                var u = u
                if u.1 > 0.5 {
                        s = 1
                        u.1 = 2 * (u.1 - 0.5)
                } else {
                        s = -1
                        u.1 = 2 * (0.5 - u.1)
                }
                let z =
                        (u.1 * (u.1 * (u.1 * 0.27385 - 0.73369) + 0.46341))
                        / (u.1 * (u.1 * (u.1 * 0.093073 + 0.309420) - 1.000000) + 0.597999)
                slope.1 = s * z * (1 + slope.0 * slope.0).squareRoot()
                assert(!slope.1.isInfinite)
                assert(!slope.1.isNaN)
                return slope
        }

        private func trowbridgeReitzSample(wi: Vector, alpha: (FloatX, FloatX), u: TwoRandomVariables)
                -> Vector
        {
                let wiStretched = normalized(Vector(x: alpha.0 * wi.x, y: alpha.1 * wi.y, z: wi.z))
                var slope = trowbridgeReitzSample11(cosTheta: cosTheta(wiStretched), u: u)
                let tmp = cosPhi(wiStretched) * slope.0 - sinPhi(wiStretched) * slope.1
                slope.1 = sinPhi(wiStretched) * slope.0 + cosPhi(wiStretched) * slope.1
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
