final class Uber: Material {

        init(
                kd: SpectrumTexture,
                ks: SpectrumTexture,
                kr: SpectrumTexture,
                kt: SpectrumTexture,
                roughness: FloatTexture,
                remapRoughness: Bool,
                opacity: SpectrumTexture,
                eta: FloatTexture
        ) {
                self.kd = kd
                self.ks = ks
                self.kr = kr
                self.kt = kt
                self.roughness = roughness
                self.remapRoughness = remapRoughness
                self.opacity = opacity
                self.eta = eta
        }

        func computeScatteringFunctions(interaction: Interaction) -> (BSDF, BSSRDF?) {
                var bsdf = BSDF(interaction: interaction)
                let eta = self.eta.evaluateFloat(at: interaction)
                let opacity = self.opacity.evaluateSpectrum(at: interaction)
                let t = -opacity + white
                if !t.isBlack {
                        bsdf.add(bxdf: SpecularTransmission(t: t, etaA: 1, etaB: 1))
                } else {
                        bsdf.eta = eta
                }
                let kd = opacity * self.kd.evaluateSpectrum(at: interaction)
                if !kd.isBlack {
                        bsdf.add(bxdf: LambertianReflection(reflectance: kd))
                }
                let ks = opacity * self.ks.evaluateSpectrum(at: interaction)
                if !ks.isBlack {
                        var roughness = self.roughness.evaluateFloat(at: interaction)
                        if remapRoughness {
                                roughness = TrowbridgeReitzDistribution.getAlpha(from: roughness)
                        }
                        let trowbridge = TrowbridgeReitzDistribution(alpha: (roughness, roughness))
                        let dielectric = FresnelDielectric(etaI: 1, etaT: eta)
                        let microfacet = MicrofacetReflection(
                                reflectance: ks, distribution: trowbridge, fresnel: dielectric)
                        bsdf.add(bxdf: microfacet)

                }
                let kr = opacity * self.kr.evaluateSpectrum(at: interaction)
                if !kr.isBlack {
                        let dielectric = FresnelDielectric(etaI: 1, etaT: eta)
                        let specular = SpecularReflection(reflectance: kr, fresnel: dielectric)
                        bsdf.add(bxdf: specular)
                }
                let kt = opacity * self.kt.evaluateSpectrum(at: interaction)
                if !kt.isBlack {
                        let transmission = SpecularTransmission(t: kt, etaA: 1, etaB: eta)
                        bsdf.add(bxdf: transmission)
                }
                return (bsdf, nil)
        }

        var kd: SpectrumTexture
        var ks: SpectrumTexture
        var kr: SpectrumTexture
        var kt: SpectrumTexture
        var roughness: FloatTexture
        let remapRoughness: Bool
        var opacity: SpectrumTexture
        var eta: FloatTexture
}

func createUber(parameters: ParameterDictionary) throws -> Uber {

        func makeSpectrum(_ name: String, else value: Spectrum) throws -> SpectrumTexture {
                return try parameters.findSpectrumTexture(name: name, else: value)
        }

        func makeFloatX(_ name: String, else value: FloatX) throws -> FloatTexture {
                return try parameters.findFloatXTexture(name: name, else: value)
        }

        let kd = try makeSpectrum("Kd", else: Spectrum(intensity: 0.25))
        let ks = try makeSpectrum("Ks", else: Spectrum(intensity: 0.25))
        let kr = try makeSpectrum("Kr", else: Spectrum(intensity: 0))
        let kt = try makeSpectrum("Kt", else: Spectrum(intensity: 0))
        let roughness = try makeFloatX("roughness", else: 0.1)
        let eta = try makeFloatX("eta", else: 1.5)
        let opacity = try makeSpectrum("opacity", else: Spectrum(intensity: 1))
        let remapRoughness = try parameters.findOneBool(called: "remapRoughness", else: true)

        return Uber(
                kd: kd, ks: ks, kr: kr, kt: kt,
                roughness: roughness, remapRoughness: remapRoughness,
                opacity: opacity, eta: eta)
}
