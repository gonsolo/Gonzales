final class Glass: Material {

        init(reflectance: SpectrumTexture, transmittance: SpectrumTexture, eta: FloatTexture) {
                self.reflectance = reflectance
                self.transmittance = transmittance
                self.eta = eta
        }

        func computeScatteringFunctions(interaction: Interaction) -> (BSDF, BSSRDF?) {
                var bsdf = BSDF(interaction: interaction)
                let eta = self.eta.evaluateFloat(at: interaction)
                bsdf.eta = eta
                let reflectance = self.reflectance.evaluateSpectrum(at: interaction)
                let transmittance = self.transmittance.evaluateSpectrum(at: interaction)
                if reflectance.isBlack && transmittance.isBlack { return (bsdf, nil) }
                let specular = FresnelSpecular(
                        reflectance: reflectance, transmittance: transmittance, etaA: 1, etaB: eta)
                bsdf.set(bxdf: specular)
                return (bsdf, nil)
        }

        var reflectance: SpectrumTexture
        var transmittance: SpectrumTexture
        var eta: FloatTexture
}

func createGlass(parameters: ParameterDictionary) throws -> Glass {
        let kr = try parameters.findSpectrumTexture(name: "Kr", else: Spectrum(intensity: 1))
        let kt = try parameters.findSpectrumTexture(name: "Kt", else: Spectrum(intensity: 1))
        let eta = try parameters.findFloatXTexture(name: "eta", else: 1.5)
        return Glass(reflectance: kr, transmittance: kt, eta: eta)
}
