struct Conductor {

        @MainActor
        func getBsdf(interaction: Interaction) -> GlobalBsdf {
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
                return microfaceReflectionBsdf
        }

        var eta: RgbSpectrum
        var k: RgbSpectrum
        var roughness: (FloatX, FloatX)
}

@MainActor
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
