final class Conductor: Material {

        init(eta: RGBSpectrum, k: RGBSpectrum, roughness: (FloatX, FloatX)) {
                self.eta = eta
                self.k = k
                self.roughness = roughness
        }

        func getGlobalBsdf(interaction: Interaction) -> GlobalBsdf {
                let alpha = (max(roughness.0, 1e-3), max(roughness.1, 1e-3))
                let trowbridge = TrowbridgeReitzDistribution(alpha: alpha)
                let fresnel = FresnelConductor(
                        etaI: white,
                        etaT: eta,
                        k: k)
                let reflection = MicrofacetReflection(
                        reflectance: white,
                        distribution: trowbridge,
                        fresnel: fresnel)
                let bsdf = GlobalBsdf(bxdf: reflection, interaction: interaction)
                return bsdf
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
