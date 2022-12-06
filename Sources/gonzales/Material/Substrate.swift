final class Substrate: Material {

        init(kd: SpectrumTexture, ks: SpectrumTexture, roughness: (FloatX, FloatX)) {
                self.kd = kd
                self.ks = ks
                self.roughness = roughness
        }

        func computeScatteringFunctions(interaction: Interaction) -> (BSDF, BSSRDF?) {
                var bsdf = BSDF(interaction: interaction)
                let kd = self.kd.evaluateSpectrum(at: interaction)
                let ks = self.ks.evaluateSpectrum(at: interaction)
                let alpha = TrowbridgeReitzDistribution.getAlpha(from: roughness)
                let distribution = TrowbridgeReitzDistribution(alpha: alpha)
                let fresnelBlend = FresnelBlend(
                        diffuseReflection: kd,
                        glossyReflection: ks,
                        distribution: distribution)
                bsdf.set(bxdf: fresnelBlend)
                return (bsdf, nil)
        }

        var kd: SpectrumTexture
        var ks: SpectrumTexture
        var roughness: (FloatX, FloatX)
        let remapRoughness = true
}

func createSubstrate(parameters: ParameterDictionary) throws -> Substrate {
        let kd: SpectrumTexture = try parameters.findSpectrumTexture(name: "Kd", else: gray)
        let ks: SpectrumTexture = try parameters.findSpectrumTexture(name: "Ks", else: gray)
        let uroughness = try parameters.findOneFloatX(called: "uroughness", else: 0.1)
        let vroughness = try parameters.findOneFloatX(called: "vroughness", else: 0.1)
        let roughness = (uroughness, vroughness)
        return Substrate(kd: kd, ks: ks, roughness: roughness)
}
