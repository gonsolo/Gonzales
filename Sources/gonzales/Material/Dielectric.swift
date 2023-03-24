final class Dielectric: Material {

        init(reflectance: RGBSpectrumTexture, transmittance: RGBSpectrumTexture, eta: FloatTexture) {
                self.reflectance = reflectance
                self.transmittance = transmittance
                self.eta = eta
        }

        func getBSDF(interaction: Interaction) -> BSDF {
                var bsdf = BSDF(interaction: interaction)
                let eta = self.eta.evaluateFloat(at: interaction)
                let reflectance = self.reflectance.evaluateRGBSpectrum(at: interaction)
                let transmittance = self.transmittance.evaluateRGBSpectrum(at: interaction)
                if reflectance.isBlack && transmittance.isBlack { return bsdf }
                let specular = FresnelSpecular(
                        reflectance: reflectance, transmittance: transmittance, etaA: 1, etaB: eta)
                bsdf.set(bxdf: specular)
                return bsdf
        }

        var reflectance: RGBSpectrumTexture
        var transmittance: RGBSpectrumTexture
        var eta: FloatTexture
}

func createDielectric(parameters: ParameterDictionary) throws -> Dielectric {
        let kr = try parameters.findRGBSpectrumTexture(name: "Kr", else: RGBSpectrum(intensity: 1))
        let kt = try parameters.findRGBSpectrumTexture(name: "Kt", else: RGBSpectrum(intensity: 1))
        let eta = try parameters.findFloatXTexture(name: "eta", else: 1.5)
        return Dielectric(reflectance: kr, transmittance: kt, eta: eta)
}
