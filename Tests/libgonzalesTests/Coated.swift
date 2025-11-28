import XCTest

@testable import libgonzales

final class ShadingFrameTests: XCTestCase {

        func runShadingTest(frame shadingFrame: ShadingFrame, description: String) {

                let bsdfFrame = BsdfFrame(geometricNormal: upNormal, shadingFrame: shadingFrame)
                let alpha: (FloatX, FloatX) = (0.001, 0.001)
                let distribution = TrowbridgeReitzDistribution(alpha: alpha)
                let dielectric = DielectricBsdf(
                        distribution: distribution, refractiveIndex: 1.0, bsdfFrame: bsdfFrame)
                let diffuse = DiffuseBsdf(reflectance: red, bsdfFrame: bsdfFrame)

                let coated = CoatedDiffuseBsdf(
                        dielectric: dielectric,
                        diffuse: diffuse,
                        bsdfFrame: bsdfFrame)

                let wo = Vector(x: 0.0, y: 0.0, z: 1.0)
                let wi = Vector(x: 0.0, y: 0.1, z: 1.0)

                let worldSpectrum = coated.evaluateWorld(wo: wo, wi: wi)

                XCTAssertTrue(
                        worldSpectrum.x > 0.2 && worldSpectrum.x < 0.4,
                        "World spectrum X component out of range for \(description)")
                XCTAssertEqual(
                        worldSpectrum.y, 0, accuracy: 1e-6,
                        "World spectrum Y should be zero for \(description)")
                XCTAssertEqual(
                        worldSpectrum.z, 0, accuracy: 1e-6,
                        "World spectrum Z should be zero for \(description)")
        }

        func testShadingFrameIdentity() throws {
                let shadingFrame1 = ShadingFrame(
                        x: Vector(x: 1, y: 0, z: 0),
                        y: Vector(x: 0, y: 1, z: 0),
                        z: Vector(x: 0, y: 0, z: 1))
                runShadingTest(frame: shadingFrame1, description: "Identity Frame")
        }

        func testShadingFrameFromXY() throws {
                let shadingFrame2 = ShadingFrame(
                        x: Vector(x: 1, y: 0, z: 0),
                        y: Vector(x: 0, y: 1, z: 0))
                runShadingTest(frame: shadingFrame2, description: "Frame from X and Y")
        }

        func testShadingFrameFromZAxis() throws {
                let shadingFrame3 = ShadingFrame(
                        z: normalized(Vector(x: 1, y: 10, z: 1)))
                runShadingTest(frame: shadingFrame3, description: "Frame from Z-axis construction")
        }
}
