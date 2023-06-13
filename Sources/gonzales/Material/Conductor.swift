final class Conductor: Material {

        init(eta: RGBSpectrum, k: RGBSpectrum, roughness: (FloatX, FloatX)) {
                self.eta = eta
                self.k = k
                self.roughness = roughness
        }

        func setBsdf(interaction: inout SurfaceInteraction) {
                let alpha = (max(roughness.0, 1e-3), max(roughness.1, 1e-3))
                let trowbridge = TrowbridgeReitzDistribution(alpha: alpha)
                let fresnel = FresnelConductor(
                        etaI: white,
                        etaT: eta,
                        k: k)
                let bsdfFrame = BsdfFrame(interaction: interaction)
                let microfaceReflectionBsdf = MicrofacetReflection(
                        reflectance: white,
                        distribution: trowbridge,
                        fresnel: fresnel,
                        bsdfFrame: bsdfFrame)
                interaction.bsdf = microfaceReflectionBsdf
        }

        var eta: RGBSpectrum
        var k: RGBSpectrum
        var roughness: (FloatX, FloatX)
}

func createConductor(parameters: ParameterDictionary) throws -> Conductor {
        let eta = try parameters.findSpectrum(name: "eta") ?? namedSpectra["metal-Cu-eta"]!
        let k = try parameters.findSpectrum(name: "k") ?? namedSpectra["metal-Cu-k"]!
        //let remapRoughness = try findOneBool(called: "remaproughness", else: false)
        let roughnessOptional = try parameters.findOneFloatXOptional(called: "roughness")
        let uRoughness =
                try roughnessOptional ?? parameters.findOneFloatX(called: "uroughness", else: 0.5)
        let vRoughness =
                try roughnessOptional ?? parameters.findOneFloatX(called: "vroughness", else: 0.5)
        let roughness = (uRoughness, vRoughness)

        let etaRgb = eta.asRgb()
        let kRgb = k.asRgb()
        return Conductor(eta: etaRgb, k: kRgb, roughness: roughness)
}
