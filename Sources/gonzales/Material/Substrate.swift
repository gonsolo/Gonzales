final class Substrate: Material {

        init(kd: RGBSpectrumTexture, ks: RGBSpectrumTexture, roughness: (FloatX, FloatX)) {
                self.kd = kd
                self.ks = ks
                self.roughness = roughness
        }

        func getBSDF(interaction: Interaction) -> BSDF {
                var bsdf = BSDF(interaction: interaction)
                let kd = self.kd.evaluateRGBSpectrum(at: interaction)
                let ks = self.ks.evaluateRGBSpectrum(at: interaction)
                let alpha = TrowbridgeReitzDistribution.getAlpha(from: roughness)
                let distribution = TrowbridgeReitzDistribution(alpha: alpha)
                let fresnelBlend = FresnelBlend(
                        diffuseReflection: kd,
                        glossyReflection: ks,
                        distribution: distribution)
                bsdf.set(bxdf: fresnelBlend)
                return bsdf
        }

        var kd: RGBSpectrumTexture
        var ks: RGBSpectrumTexture
        var roughness: (FloatX, FloatX)
        let remapRoughness = true
}

func createSubstrate(parameters: ParameterDictionary) throws -> Substrate {
        let kd: RGBSpectrumTexture = try parameters.findRGBSpectrumTexture(name: "Kd", else: gray)
        let ks: RGBSpectrumTexture = try parameters.findRGBSpectrumTexture(name: "Ks", else: gray)
        let uroughness = try parameters.findOneFloatX(called: "uroughness", else: 0.1)
        let vroughness = try parameters.findOneFloatX(called: "vroughness", else: 0.1)
        let roughness = (uroughness, vroughness)
        return Substrate(kd: kd, ks: ks, roughness: roughness)
}
