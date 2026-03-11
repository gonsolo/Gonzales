import XCTest

@testable import libgonzales

final class FresnelTests: XCTestCase {

        // MARK: - Normal Incidence

        func testNormalIncidenceGlass() {
                // At normal incidence (cosθ=1) with glass (η=1.5):
                // R = ((1 - 1.5) / (1 + 1.5))^2 = (-0.5 / 2.5)^2 = 0.04
                let result = FresnelDielectric.reflected(cosThetaI: 1.0, refractiveIndex: 1.5)
                XCTAssertEqual(result, 0.04, accuracy: 1e-4)
        }

        func testNormalIncidenceWater() {
                // Water η≈1.33: R = ((1 - 1.33) / (1 + 1.33))^2 = 0.02
                let result = FresnelDielectric.reflected(cosThetaI: 1.0, refractiveIndex: 1.33)
                let expected: FloatX = pow((1 - 1.33) / (1 + 1.33), 2)
                XCTAssertEqual(result, expected, accuracy: 1e-4)
        }

        // MARK: - No Boundary (η = 1)

        func testNoBoundary() {
                // Same medium on both sides => no reflection
                let result = FresnelDielectric.reflected(cosThetaI: 1.0, refractiveIndex: 1.0)
                XCTAssertEqual(result, 0.0, accuracy: 1e-6)
        }

        func testNoBoundaryAtAngle() {
                let result = FresnelDielectric.reflected(cosThetaI: 0.707, refractiveIndex: 1.0)
                XCTAssertEqual(result, 0.0, accuracy: 1e-5)
        }

        // MARK: - Grazing Incidence

        func testGrazingIncidence() {
                // At grazing angle, reflectance approaches 1
                let result = FresnelDielectric.reflected(cosThetaI: 0.001, refractiveIndex: 1.5)
                XCTAssertGreaterThan(result, 0.95)
        }

        // MARK: - Total Internal Reflection

        func testTotalInternalReflection() {
                // Going from glass (η=1.5) to air at steep angle
                // sin²θ_t = sin²θ_i / η² > 1 for large enough angles
                // Critical angle for glass: sin(θ_c) = 1/1.5 ≈ 0.667, cos(θ_c) ≈ 0.745
                // At angle steeper than critical (cosθ < 0.745), TIR occurs
                // Negative cosθ means we're going from the denser medium
                let result = FresnelDielectric.reflected(cosThetaI: -0.3, refractiveIndex: 1.5)
                XCTAssertEqual(result, 1.0, accuracy: 1e-6)
        }

        // MARK: - Symmetry and Range

        func testResultAlwaysInZeroOneRange() {
                let angles: [FloatX] = [-1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 0.7, 0.9, 1.0]
                let etas: [FloatX] = [1.0, 1.33, 1.5, 2.0, 2.5]
                for cosTheta in angles {
                        for eta in etas {
                                let result = FresnelDielectric.reflected(
                                        cosThetaI: cosTheta, refractiveIndex: eta)
                                XCTAssertGreaterThanOrEqual(
                                        result, 0,
                                        "Negative reflectance for cosθ=\(cosTheta), η=\(eta)")
                                XCTAssertLessThanOrEqual(
                                        result, 1.0 + 1e-6,
                                        "Reflectance > 1 for cosθ=\(cosTheta), η=\(eta)")
                        }
                }
        }

        func testReflectanceIncreasesWithAngle() {
                // Reflectance should increase monotonically as cosθ decreases from 1 to 0
                let eta: FloatX = 1.5
                var previousReflectance: FloatX = 0
                for i in stride(from: 10, through: 1, by: -1) {
                        let cosTheta = FloatX(i) / 10.0
                        let reflectance = FresnelDielectric.reflected(
                                cosThetaI: cosTheta, refractiveIndex: eta)
                        XCTAssertGreaterThanOrEqual(
                                reflectance, previousReflectance - 1e-6,
                                "Reflectance not monotonically increasing at cosθ=\(cosTheta)")
                        previousReflectance = reflectance
                }
        }

        // MARK: - Clamping

        func testCosThetaClamping() {
                // Values outside [-1, 1] should be clamped
                let result1 = FresnelDielectric.reflected(cosThetaI: 1.5, refractiveIndex: 1.5)
                let result2 = FresnelDielectric.reflected(cosThetaI: 1.0, refractiveIndex: 1.5)
                XCTAssertEqual(result1, result2, accuracy: 1e-6)
        }
}
