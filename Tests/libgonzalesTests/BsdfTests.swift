import Testing

@testable import libgonzales

@Test("Evaluate CoatedDiffuseBsdf mathematically")
func coatedDiffuseEvaluate() throws {
        let shadingFrame = ShadingFrame(
                x: Vector(x: 1, y: 0, z: 0),
                y: Vector(x: 0, y: 1, z: 0),
                z: Vector(x: 0, y: 0, z: 1)
        )
        let normal = Normal(x: 0, y: 0, z: 1)
        let bsdfFrame = BsdfFrame(geometricNormal: normal, shadingFrame: shadingFrame)

        let top = DielectricBsdf(
                distribution: TrowbridgeReitzDistribution(alpha: (0.001, 0.001)),
                refractiveIndex: 1.5,
                bsdfFrame: bsdfFrame
        )

        let bottom = DiffuseBsdf(
                reflectance: RgbSpectrum(intensity: 0.8),
                bsdfFrame: bsdfFrame
        )

        let coated = CoatedDiffuseBsdf(
                dielectric: top,
                diffuse: bottom,
                thickness: 0.1,
                albedo: RgbSpectrum(intensity: 1.0),
                asymmetry: 0.0,
                maxDepth: 10,
                nSamples: 1,
                bsdfFrame: bsdfFrame
        )

        let outVec = Vector(x: 0, y: 0.94868, z: 0.31622)  // Grazing viewing angle
        let inVec = Vector(x: 0, y: 0.0, z: 1.0)  // Light from directly above

        let f = coated.evaluate(outgoing: outVec, incident: inVec)
        let pdf = coated.probabilityDensity(outgoing: outVec, incident: inVec)

        print("====== BSDF TEST ======")
        let fDiffuse = bottom.evaluate(outgoing: outVec, incident: inVec)
        print("Diffuse evaluate: \(fDiffuse)")
        print("Layered evaluate: \(f)")
        print("Layered PDF: \(pdf)")

        let sampleDiffuse = bottom.sample(outgoing: outVec, uSample: (0.5, 0.5, 0.5))
        print("Diffuse sample estimate: \(sampleDiffuse.estimate)")

        let sample = coated.sample(outgoing: outVec, uSample: (0.5, 0.5, 0.5))
        print("Sample estimate: \(sample.estimate)")
        print("Sample PDF: \(sample.probabilityDensity)")
        print("Sample incoming: \(sample.incoming)")
        print("=======================")
}
