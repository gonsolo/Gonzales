struct Conductor {

        func getBsdf(interaction: any Interaction, arena _: TextureArena) -> MicrofacetReflection {
                let alpha = (max(roughness.0, 1e-3), max(roughness.1, 1e-3))
                let trowbridge = TrowbridgeReitzDistribution(alpha: alpha)
                let fresnel = FresnelConductor(
                        etaI: white,
                        etaT: eta,
                        extinction: extinction)
                let bsdfFrame = BsdfFrame(interaction: interaction)
                let microfaceReflectionBsdf = MicrofacetReflection(
                        reflectance: white,
                        distribution: trowbridge,
                        fresnel: fresnel,
                        bsdfFrame: bsdfFrame)
                return microfaceReflectionBsdf
        }

        var eta: RgbSpectrum
        var extinction: RgbSpectrum
        var roughness: (Real, Real)
}

extension Conductor {
        static func create(parameters: ParameterDictionary, arena _: inout TextureArena) throws -> Conductor {
                let eta = try parameters.findSpectrum(name: "eta") ?? namedSpectra["metal-Cu-eta"]!
                let extinctionParameter =
                        try parameters.findSpectrum(name: "k") ?? namedSpectra["metal-Cu-k"]!
                // let remapRoughness = try findOneBool(called: "remaproughness", else: false)
                let roughnessOptional = try parameters.findOneRealOptional(called: "roughness")
                let uRoughness =
                        try roughnessOptional ?? parameters.findOneReal(called: "uroughness", else: 0.5)
                let vRoughness =
                        try roughnessOptional ?? parameters.findOneReal(called: "vroughness", else: 0.5)
                let roughness = (uRoughness, vRoughness)

                let etaRgb = eta.asRgb()
                let extinctionRgb = extinctionParameter.asRgb()
                return Conductor(eta: etaRgb, extinction: extinctionRgb, roughness: roughness)
        }
}
