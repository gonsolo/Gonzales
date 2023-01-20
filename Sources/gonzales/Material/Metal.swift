final class Metal: Material {

        init(eta: RGBSpectrum, k: RGBSpectrum, roughness: (FloatX, FloatX)) {
                self.eta = eta
                self.k = k
                self.roughness = roughness
        }

        func computeScatteringFunctions(interaction: Interaction) -> BSDF {
                var bsdf = BSDF(interaction: interaction)
                let trowbridge = TrowbridgeReitzDistribution(alpha: roughness)
                let fresnel = FresnelConductor(
                        etaI: white,
                        etaT: eta,
                        k: k)
                let reflection = MicrofacetReflection(
                        reflectance: white,
                        distribution: trowbridge,
                        fresnel: fresnel)
                bsdf.set(bxdf: reflection)
                return bsdf
        }

        var eta: RGBSpectrum
        var k: RGBSpectrum
        var roughness: (FloatX, FloatX)
}

func createMetal(parameters: ParameterDictionary) throws -> Metal {
        guard let eta = try parameters.findSpectrum(name: "eta") as? RGBSpectrum else {
                throw ParameterError.missing(parameter: "eta")
        }
        guard let k = try parameters.findSpectrum(name: "k") as? RGBSpectrum else {
                throw ParameterError.missing(parameter: "k")
        }
        // ignored let remapRoughness = try findOneBool(called : "remaproughness", else: false)
        let uRoughness = try parameters.findOneFloatX(called: "uroughness", else: 0.5)
        let vRoughness = try parameters.findOneFloatX(called: "vroughness", else: 0.5)
        let roughness = (uRoughness, vRoughness)
        return Metal(eta: eta, k: k, roughness: roughness)
}
