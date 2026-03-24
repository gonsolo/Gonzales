import Testing

@testable import libgonzales

// Helper to create an identity BsdfFrame
private let identityFrame: BsdfFrame = {
        let shadingFrame = ShadingFrame(
                x: Vector(x: 1, y: 0, z: 0),
                y: Vector(x: 0, y: 1, z: 0),
                z: Vector(x: 0, y: 0, z: 1))
        return BsdfFrame(geometricNormal: Normal(x: 0, y: 0, z: 1), shadingFrame: shadingFrame)
}()

// MARK: - DiffuseBsdf Tests

@Suite struct DiffuseBsdfTests {

        @Test func evaluateReturnsReflectanceOverPi() {
                let reflectance = RgbSpectrum(red: 0.8, green: 0.4, blue: 0.2)
                let bsdf = DiffuseBsdf(reflectance: reflectance, bsdfFrame: identityFrame)
                let outgoing = Vector(x: 0, y: 0, z: 1)
                let incident = normalized(Vector(x: 0.5, y: 0, z: 1))
                let result = bsdf.evaluate(outgoing: outgoing, incident: incident)
                #expect(abs(result.red - reflectance.red / Real.pi) <= 1e-5)
                #expect(abs(result.green - reflectance.green / Real.pi) <= 1e-5)
                #expect(abs(result.blue - reflectance.blue / Real.pi) <= 1e-5)
        }

        @Test func evaluateBlackReflectanceReturnsBlack() {
                let bsdf = DiffuseBsdf(reflectance: black, bsdfFrame: identityFrame)
                let result = bsdf.evaluate(
                        outgoing: Vector(x: 0, y: 0, z: 1),
                        incident: Vector(x: 0, y: 0, z: 1))
                #expect(result.isBlack)
        }

        @Test func diffuseIsNotSpecular() {
                let bsdf = DiffuseBsdf(reflectance: white, bsdfFrame: identityFrame)
                #expect(!bsdf.isSpecular)
        }

        @Test func diffuseAlbedoReturnsReflectance() {
                let reflectance = RgbSpectrum(red: 0.5, green: 0.6, blue: 0.7)
                let bsdf = DiffuseBsdf(reflectance: reflectance, bsdfFrame: identityFrame)
                let albedo = bsdf.albedo()
                #expect(abs(albedo.red - reflectance.red) <= 1e-6)
                #expect(abs(albedo.green - reflectance.green) <= 1e-6)
                #expect(abs(albedo.blue - reflectance.blue) <= 1e-6)
        }

        @Test func diffuseSampleProducesUpperHemisphere() {
                let bsdf = DiffuseBsdf(reflectance: white, bsdfFrame: identityFrame)
                let outgoing = Vector(x: 0, y: 0, z: 1)
                let samples: [ThreeRandomVariables] = [
                        (0.1, 0.2, 0.3), (0.5, 0.5, 0.5), (0.9, 0.1, 0.8),
                ]
                for uSample in samples {
                        let sample = bsdf.sample(outgoing: outgoing, uSample: uSample)
                        #expect(sample.incoming.z >= 0, "Sampled direction should be in upper hemisphere")
                }
        }

        @Test func diffuseSamplePdfPositive() {
                let bsdf = DiffuseBsdf(reflectance: white, bsdfFrame: identityFrame)
                let outgoing = Vector(x: 0, y: 0, z: 1)
                let sample = bsdf.sample(outgoing: outgoing, uSample: (0.5, 0.5, 0.5))
                #expect(sample.probabilityDensity > 0)
        }

        @Test func diffuseEnergyConservation() {
                let bsdf = DiffuseBsdf(reflectance: white, bsdfFrame: identityFrame)
                let outgoing = Vector(x: 0, y: 0, z: 1)
                // Monte Carlo integration: ∫ f(wo,wi) |cosθ| dωi should be ≤ 1
                var sum: Real = 0
                let numSamples = 200
                for i in 0..<numSamples {
                        let u1 = Real(i) / Real(numSamples)
                        let u2 = Real((i * 7 + 3) % numSamples) / Real(numSamples)
                        let sample = bsdf.sample(outgoing: outgoing, uSample: (u1, u2, 0.5))
                        if sample.probabilityDensity > 0 {
                                let f = bsdf.evaluate(outgoing: outgoing, incident: sample.incoming)
                                let cosTheta = absCosTheta(sample.incoming)
                                sum += f.red * cosTheta / sample.probabilityDensity
                        }
                }
                let estimate = sum / Real(numSamples)
                #expect(estimate <= 1.0 + 0.1, "Energy conservation violated: \(estimate)")
        }
}

// MARK: - DielectricBsdf Tests

@Suite struct DielectricBsdfTests {

        @Test func dielectricEtaOneReturnsBlack() {
                let dist = TrowbridgeReitzDistribution(alpha: (0.5, 0.5))
                let bsdf = DielectricBsdf(
                        distribution: dist, refractiveIndex: 1.0, bsdfFrame: identityFrame)
                let result = bsdf.evaluate(
                        outgoing: Vector(x: 0, y: 0, z: 1),
                        incident: normalized(Vector(x: 0.3, y: 0, z: 1)))
                #expect(result.isBlack, "No boundary (eta=1) should return black")
        }

        @Test func dielectricSmoothIsSpecular() {
                let dist = TrowbridgeReitzDistribution(alpha: (0.0005, 0.0005))
                let bsdf = DielectricBsdf(
                        distribution: dist, refractiveIndex: 1.5, bsdfFrame: identityFrame)
                #expect(bsdf.isSpecular)
        }

        @Test func dielectricRoughIsNotSpecular() {
                let dist = TrowbridgeReitzDistribution(alpha: (0.5, 0.5))
                let bsdf = DielectricBsdf(
                        distribution: dist, refractiveIndex: 1.5, bsdfFrame: identityFrame)
                #expect(!bsdf.isSpecular)
        }

        @Test func dielectricSpecularSamplePdfPositive() {
                let dist = TrowbridgeReitzDistribution(alpha: (0.0005, 0.0005))
                let bsdf = DielectricBsdf(
                        distribution: dist, refractiveIndex: 1.5, bsdfFrame: identityFrame)
                let outgoing = Vector(x: 0, y: 0, z: 1)
                let sample = bsdf.sample(outgoing: outgoing, uSample: (0.1, 0.5, 0.5))
                // For specular, either reflect or transmit — pdf should be non-zero
                #expect(sample.probabilityDensity > 0)
        }

        @Test func dielectricSampleEitherReflectsOrRefracts() {
                let dist = TrowbridgeReitzDistribution(alpha: (0.0005, 0.0005))
                let bsdf = DielectricBsdf(
                        distribution: dist, refractiveIndex: 1.5, bsdfFrame: identityFrame)
                let outgoing = Vector(x: 0, y: 0, z: 1)
                // With u0 < probabilityReflected → reflection (z > 0)
                let reflectSample = bsdf.sample(outgoing: outgoing, uSample: (0.01, 0.5, 0.5))
                #expect(reflectSample.incoming.z > 0, "Should reflect for small u0")
                // With u0 > probabilityReflected → transmission (z < 0)
                let transmitSample = bsdf.sample(outgoing: outgoing, uSample: (0.99, 0.5, 0.5))
                #expect(transmitSample.incoming.z < 0, "Should transmit for large u0")
        }

        @Test func dielectricAlbedoIsWhite() {
                let dist = TrowbridgeReitzDistribution(alpha: (0.5, 0.5))
                let bsdf = DielectricBsdf(
                        distribution: dist, refractiveIndex: 1.5, bsdfFrame: identityFrame)
                let albedo = bsdf.albedo()
                #expect(abs(albedo.red - 1.0) <= 1e-6)
                #expect(abs(albedo.green - 1.0) <= 1e-6)
                #expect(abs(albedo.blue - 1.0) <= 1e-6)
        }
}

// MARK: - MixBsdf Tests

@Suite struct MixBsdfTests {

        private func makeDiffuse(reflectance: RgbSpectrum) -> BsdfVariant {
                return .diffuseBsdf(DiffuseBsdf(reflectance: reflectance, bsdfFrame: identityFrame))
        }

        @Test func mixAmount0ReturnsBsdf1Only() {
                let bsdf1 = makeDiffuse(reflectance: RgbSpectrum(intensity: 0.8))
                let bsdf2 = makeDiffuse(reflectance: RgbSpectrum(intensity: 0.2))
                let mix = MixBsdf(bsdf1: bsdf1, bsdf2: bsdf2, amount: 0.0, bsdfFrame: identityFrame)
                let outgoing = Vector(x: 0, y: 0, z: 1)
                let incident = normalized(Vector(x: 0.5, y: 0, z: 1))
                let mixResult = mix.evaluate(outgoing: outgoing, incident: incident)
                let bsdf1Result = bsdf1.evaluate(outgoing: outgoing, incident: incident)
                #expect(abs(mixResult.red - bsdf1Result.red) <= 1e-5)
        }

        @Test func mixAmount1ReturnsBsdf2Only() {
                let bsdf1 = makeDiffuse(reflectance: RgbSpectrum(intensity: 0.8))
                let bsdf2 = makeDiffuse(reflectance: RgbSpectrum(intensity: 0.2))
                let mix = MixBsdf(bsdf1: bsdf1, bsdf2: bsdf2, amount: 1.0, bsdfFrame: identityFrame)
                let outgoing = Vector(x: 0, y: 0, z: 1)
                let incident = normalized(Vector(x: 0.5, y: 0, z: 1))
                let mixResult = mix.evaluate(outgoing: outgoing, incident: incident)
                let bsdf2Result = bsdf2.evaluate(outgoing: outgoing, incident: incident)
                #expect(abs(mixResult.red - bsdf2Result.red) <= 1e-5)
        }

        @Test func mixAmountHalfBlends() {
                let reflectance1 = RgbSpectrum(intensity: 1.0)
                let reflectance2 = RgbSpectrum(intensity: 0.0)
                let bsdf1 = makeDiffuse(reflectance: reflectance1)
                let bsdf2 = makeDiffuse(reflectance: reflectance2)
                let mix = MixBsdf(bsdf1: bsdf1, bsdf2: bsdf2, amount: 0.5, bsdfFrame: identityFrame)
                let outgoing = Vector(x: 0, y: 0, z: 1)
                let incident = normalized(Vector(x: 0.3, y: 0, z: 1))
                let result = mix.evaluate(outgoing: outgoing, incident: incident)
                let half = bsdf1.evaluate(outgoing: outgoing, incident: incident)
                // result ≈ 0.5 * bsdf1 + 0.5 * 0 = 0.5 * bsdf1
                #expect(abs(result.red - half.red * 0.5) <= 1e-4)
        }

        @Test func mixPdfIsWeightedAverage() {
                let bsdf1 = makeDiffuse(reflectance: white)
                let bsdf2 = makeDiffuse(reflectance: white)
                let amount: Real = 0.3
                let mix = MixBsdf(bsdf1: bsdf1, bsdf2: bsdf2, amount: amount, bsdfFrame: identityFrame)
                let outgoing = Vector(x: 0, y: 0, z: 1)
                let incident = normalized(Vector(x: 0.3, y: 0, z: 1))
                let mixPdf = mix.probabilityDensity(outgoing: outgoing, incident: incident)
                let pdf1 = bsdf1.probabilityDensity(outgoing: outgoing, incident: incident)
                let pdf2 = bsdf2.probabilityDensity(outgoing: outgoing, incident: incident)
                let expected = (1.0 - amount) * pdf1 + amount * pdf2
                #expect(abs(mixPdf - expected) <= 1e-4)
        }

        @Test func mixAlbedoBlend() {
                let bsdf1 = makeDiffuse(reflectance: RgbSpectrum(red: 1, green: 0, blue: 0))
                let bsdf2 = makeDiffuse(reflectance: RgbSpectrum(red: 0, green: 0, blue: 1))
                let mix = MixBsdf(bsdf1: bsdf1, bsdf2: bsdf2, amount: 0.5, bsdfFrame: identityFrame)
                let albedo = mix.albedo()
                #expect(abs(albedo.red - 0.5) <= 1e-5)
                #expect(abs(albedo.blue - 0.5) <= 1e-5)
                #expect(abs(albedo.green) <= 1e-5)
        }
}
