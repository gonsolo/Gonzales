import Testing

@testable import libgonzales

@Suite struct ShadingFrameTests {

        func runShadingTest(frame shadingFrame: ShadingFrame, description: String) {

                let bsdfFrame = BsdfFrame(geometricNormal: upNormal, shadingFrame: shadingFrame)
                let alpha: (Real, Real) = (0.001, 0.001)
                let distribution = TrowbridgeReitzDistribution(alpha: alpha)
                let dielectric = DielectricBsdf(
                        distribution: distribution, refractiveIndex: 1.0, bsdfFrame: bsdfFrame)
                let diffuse = DiffuseBsdf(reflectance: red, bsdfFrame: bsdfFrame)

                let coated = CoatedDiffuseBsdf(
                        dielectric: dielectric,
                        diffuse: diffuse,
                        thickness: 0.0,
                        albedo: RgbSpectrum(intensity: 0.0),
                        asymmetry: 0.0,
                        maxDepth: 10,
                        nSamples: 1,
                        bsdfFrame: bsdfFrame)

                let outgoing = Vector(x: 0.0, y: 0.0, z: 1.0)
                let incident = Vector(x: 0.0, y: 0.1, z: 1.0)

                let worldSpectrum = coated.evaluateWorldSpace(outgoing: outgoing, incident: incident)

                #expect(
                        worldSpectrum.x > 0.2 && worldSpectrum.x < 0.4,
                        "World spectrum X component out of range for \(description)")
                #expect(
                        abs(worldSpectrum.y) <= 1e-6,
                        "World spectrum Y should be zero for \(description)")
                #expect(
                        abs(worldSpectrum.z) <= 1e-6,
                        "World spectrum Z should be zero for \(description)")
        }

        @Test func shadingFrameIdentity() throws {
                let shadingFrame1 = ShadingFrame(
                        x: Vector(x: 1, y: 0, z: 0),
                        y: Vector(x: 0, y: 1, z: 0),
                        z: Vector(x: 0, y: 0, z: 1))
                runShadingTest(frame: shadingFrame1, description: "Identity Frame")
        }

        @Test func shadingFrameFromXY() throws {
                let shadingFrame2 = ShadingFrame(
                        x: Vector(x: 1, y: 0, z: 0),
                        y: Vector(x: 0, y: 1, z: 0))
                runShadingTest(frame: shadingFrame2, description: "Frame from X and Y")
        }

        @Test func shadingFrameFromZAxis() throws {
                let shadingFrame3 = ShadingFrame(
                        z: normalized(Vector(x: 1, y: 10, z: 1)))
                runShadingTest(frame: shadingFrame3, description: "Frame from Z-axis construction")
        }

        // MARK: - Direct ShadingFrame Unit Tests

        @Test func worldToLocalRoundTrip() {
                let frame = ShadingFrame(z: normalized(Vector(x: 1, y: 2, z: 3)))
                let original = normalized(Vector(x: 0.5, y: 0.7, z: 0.3))
                let local = frame.worldToLocal(world: original)
                let recovered = frame.localToWorld(local: local)
                let recoveredNorm = normalized(recovered)
                #expect(abs(recoveredNorm.x - original.x) <= 1e-4)
                #expect(abs(recoveredNorm.y - original.y) <= 1e-4)
                #expect(abs(recoveredNorm.z - original.z) <= 1e-4)
        }

        @Test func localToWorldPreservesLength() {
                let frame = ShadingFrame(z: normalized(Vector(x: 0, y: 1, z: 1)))
                let local = normalized(Vector(x: 0.3, y: 0.4, z: 0.5))
                let world = frame.localToWorld(local: local)
                let len = length(world)
                #expect(abs(len - 1.0) <= 1e-4, "Transformed vector length \(len) != 1.0")
        }

        @Test func worldToLocalIdentityFrameIsIdentity() {
                let frame = ShadingFrame()
                let v = normalized(Vector(x: 1, y: 2, z: 3))
                let local = frame.worldToLocal(world: v)
                #expect(abs(local.x - v.x) <= 1e-4)
                #expect(abs(local.y - v.y) <= 1e-4)
                #expect(abs(local.z - v.z) <= 1e-4)
        }

        @Test func constructFromZAxisProducesOrthonormal() {
                let frame = ShadingFrame(z: normalized(Vector(x: 1, y: 2, z: 3)))
                // Orthogonality
                #expect(abs(dot(frame.x, frame.y)) <= 1e-4)
                #expect(abs(dot(frame.x, frame.z)) <= 1e-4)
                #expect(abs(dot(frame.y, frame.z)) <= 1e-4)
                // Unit length
                #expect(abs(length(frame.x) - 1.0) <= 1e-4)
                #expect(abs(length(frame.y) - 1.0) <= 1e-4)
                #expect(abs(length(frame.z) - 1.0) <= 1e-4)
        }
}
