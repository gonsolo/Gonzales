import Glibc
import Testing

@testable import libgonzales

@Suite struct FresnelTests {

        // MARK: - Normal Incidence

        @Test func normalIncidenceGlass() {
                // At normal incidence (cosθ=1) with glass (η=1.5):
                // R = ((1 - 1.5) / (1 + 1.5))^2 = (-0.5 / 2.5)^2 = 0.04
                let result = FresnelDielectric.reflected(cosThetaI: 1.0, refractiveIndex: 1.5)
                #expect(abs(result - 0.04) <= 1e-4)
        }

        @Test func normalIncidenceWater() {
                // Water η≈1.33: R = ((1 - 1.33) / (1 + 1.33))^2 = 0.02
                let result = FresnelDielectric.reflected(cosThetaI: 1.0, refractiveIndex: 1.33)
                let expected: FloatX = Glibc.powf(FloatX((1 - 1.33) / (1 + 1.33)), FloatX(2))
                #expect(abs(result - expected) <= 1e-4)
        }

        // MARK: - No Boundary (η = 1)

        @Test func noBoundary() {
                // Same medium on both sides => no reflection
                let result = FresnelDielectric.reflected(cosThetaI: 1.0, refractiveIndex: 1.0)
                #expect(abs(result) <= 1e-6)
        }

        @Test func noBoundaryAtAngle() {
                let result = FresnelDielectric.reflected(cosThetaI: 0.707, refractiveIndex: 1.0)
                #expect(abs(result) <= 1e-5)
        }

        // MARK: - Grazing Incidence

        @Test func grazingIncidence() {
                // At grazing angle, reflectance approaches 1
                let result = FresnelDielectric.reflected(cosThetaI: 0.001, refractiveIndex: 1.5)
                #expect(result > 0.95)
        }

        // MARK: - Total Internal Reflection

        @Test func totalInternalReflection() {
                // Going from glass (η=1.5) to air at steep angle
                // Critical angle for glass: sin(θ_c) = 1/1.5 ≈ 0.667, cos(θ_c) ≈ 0.745
                // At angle steeper than critical (cosθ < 0.745), TIR occurs
                // Negative cosθ means we're going from the denser medium
                let result = FresnelDielectric.reflected(cosThetaI: -0.3, refractiveIndex: 1.5)
                #expect(abs(result - 1.0) <= 1e-6)
        }

        // MARK: - Symmetry and Range

        @Test func resultAlwaysInZeroOneRange() {
                let angles: [FloatX] = [-1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 0.7, 0.9, 1.0]
                let etas: [FloatX] = [1.0, 1.33, 1.5, 2.0, 2.5]
                for cosTheta in angles {
                        for eta in etas {
                                let result = FresnelDielectric.reflected(
                                        cosThetaI: cosTheta, refractiveIndex: eta)
                                #expect(
                                        result >= 0,
                                        "Negative reflectance for cosθ=\(cosTheta), η=\(eta)")
                                #expect(
                                        result <= 1.0 + 1e-6,
                                        "Reflectance > 1 for cosθ=\(cosTheta), η=\(eta)")
                        }
                }
        }

        @Test func reflectanceIncreasesWithAngle() {
                // Reflectance should increase monotonically as cosθ decreases from 1 to 0
                let eta: FloatX = 1.5
                var previousReflectance: FloatX = 0
                for i in stride(from: 10, through: 1, by: -1) {
                        let cosTheta = FloatX(i) / 10.0
                        let reflectance = FresnelDielectric.reflected(
                                cosThetaI: cosTheta, refractiveIndex: eta)
                        #expect(
                                reflectance >= previousReflectance - 1e-6,
                                "Reflectance not monotonically increasing at cosθ=\(cosTheta)")
                        previousReflectance = reflectance
                }
        }

        // MARK: - Clamping

        @Test func cosThetaClamping() {
                // Values outside [-1, 1] should be clamped
                let result1 = FresnelDielectric.reflected(cosThetaI: 1.5, refractiveIndex: 1.5)
                let result2 = FresnelDielectric.reflected(cosThetaI: 1.0, refractiveIndex: 1.5)
                #expect(abs(result1 - result2) <= 1e-6)
        }
}
