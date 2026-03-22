struct CoatedConductor {

        init(
                interfaceRoughness: (Real, Real),
                conductorRoughness: (Real, Real),
                reflectance: RgbSpectrumTexture,
                refractiveIndex: FloatTexture,
                thickness: FloatTexture,
                albedo: RgbSpectrumTexture,
                asymmetry: FloatTexture,
                eta: RgbSpectrum,
                extinction: RgbSpectrum,
                maxDepth: Int,
                nSamples: Int,
                remapRoughness: Bool
        ) {
                self.interfaceRoughness = interfaceRoughness
                self.conductorRoughness = conductorRoughness
                self.reflectance = reflectance
                self.refractiveIndex = refractiveIndex
                self.thickness = thickness
                self.albedo = albedo
                self.asymmetry = asymmetry
                self.eta = eta
                self.extinction = extinction
                self.maxDepth = maxDepth
                self.nSamples = nSamples
                self.remapRoughness = remapRoughness
        }

        func getBsdf(interaction: any Interaction) -> CoatedConductorBsdf {
                let refractiveIndexValue = self.refractiveIndex.evaluateFloat(at: interaction)
                let reflectanceValue = self.reflectance.evaluateRgbSpectrum(at: interaction)
                let thicknessValue = self.thickness.evaluateFloat(at: interaction)
                let albedoValue = self.albedo.evaluateRgbSpectrum(at: interaction)
                let asymmetryValue = self.asymmetry.evaluateFloat(at: interaction)
                let bsdfFrame = BsdfFrame(interaction: interaction)

                var alphaInterface: (Real, Real) = interfaceRoughness
                var alphaConductor: (Real, Real) = conductorRoughness
                
                if remapRoughness {
                        alphaInterface = TrowbridgeReitzDistribution.getAlpha(from: interfaceRoughness)
                        alphaConductor = TrowbridgeReitzDistribution.getAlpha(from: conductorRoughness)
                }
                
                alphaInterface.0 = max(alphaInterface.0, 0.001)
                alphaInterface.1 = max(alphaInterface.1, 0.001)
                alphaConductor.0 = max(alphaConductor.0, 0.001)
                alphaConductor.1 = max(alphaConductor.1, 0.001)
                
                let distributionInterface = TrowbridgeReitzDistribution(alpha: alphaInterface)
                let dielectric = DielectricBsdf(
                        distribution: distributionInterface, refractiveIndex: refractiveIndexValue, bsdfFrame: bsdfFrame)
                        
                let distributionConductor = TrowbridgeReitzDistribution(alpha: alphaConductor)
                let fresnel = FresnelConductor(etaI: white, etaT: eta, extinction: extinction)
                let conductorBsdf = MicrofacetReflection(
                        reflectance: reflectanceValue,
                        distribution: distributionConductor,
                        fresnel: fresnel,
                        bsdfFrame: bsdfFrame)

                let coatedConductorBsdf = CoatedConductorBsdf(
                        dielectric: dielectric,
                        conductor: conductorBsdf,
                        thickness: Real(thicknessValue),
                        albedo: albedoValue,
                        asymmetry: Real(asymmetryValue),
                        maxDepth: maxDepth,
                        nSamples: nSamples,
                        bsdfFrame: bsdfFrame)
                return coatedConductorBsdf
        }

        var reflectance: RgbSpectrumTexture
        var refractiveIndex: FloatTexture
        var interfaceRoughness: (Real, Real)
        var conductorRoughness: (Real, Real)
        var thickness: FloatTexture
        var albedo: RgbSpectrumTexture
        var asymmetry: FloatTexture
        var eta: RgbSpectrum
        var extinction: RgbSpectrum
        var maxDepth: Int
        var nSamples: Int
        var remapRoughness: Bool
}

extension CoatedConductor {
        static func create(parameters: ParameterDictionary, textures: [String: Texture]) throws -> CoatedConductor {
                let remapRoughness = try parameters.findOneBool(called: "remaproughness", else: true)
                
                // Default interface roughness mapping
                let interfaceRoughnessOptional = try parameters.findOneRealOptional(called: "interface.roughness")
                        ?? parameters.findOneRealOptional(called: "roughness")
                let uInterface = try interfaceRoughnessOptional ?? parameters.findOneReal(called: "uroughness", else: 0.5)
                let vInterface = try interfaceRoughnessOptional ?? parameters.findOneReal(called: "vroughness", else: 0.5)
                let interfaceRoughness = (uInterface, vInterface)
                
                // Conductor roughness mapping
                let conductorRoughnessOptional = try parameters.findOneRealOptional(called: "conductor.roughness")
                        ?? parameters.findOneRealOptional(called: "roughness")
                let uConductor = try conductorRoughnessOptional ?? parameters.findOneReal(called: "uroughness", else: 0.5)
                let vConductor = try conductorRoughnessOptional ?? parameters.findOneReal(called: "vroughness", else: 0.5)
                let conductorRoughness = (uConductor, vConductor)

                let reflectance = try parameters.findRgbSpectrumTexture(name: "reflectance", textures: textures)
                let refractiveIndex = try parameters.findRealTexture(name: "eta", textures: textures, else: 1.5)
                
                let thickness = try parameters.findRealTexture(name: "thickness", textures: textures, else: 0.01)
                let asymmetry = try parameters.findRealTexture(name: "g", textures: textures, else: 0.0)
                let maxDepth = try parameters.findOneInt(called: "maxdepth", else: 10)
                let nSamples = try parameters.findOneInt(called: "nsamples", else: 1)
                
                let albedo = try parameters.findRgbSpectrumTexture(name: "albedo", textures: textures, else: RgbSpectrum(intensity: 0.0))
                let eta = try parameters.findSpectrum(name: "conductor.eta") ?? namedSpectra["metal-Cu-eta"]!
                let extinctionParameter = try parameters.findSpectrum(name: "conductor.k") ?? namedSpectra["metal-Cu-k"]!

                return CoatedConductor(
                        interfaceRoughness: interfaceRoughness,
                        conductorRoughness: conductorRoughness,
                        reflectance: reflectance,
                        refractiveIndex: refractiveIndex,
                        thickness: thickness,
                        albedo: albedo,
                        asymmetry: asymmetry,
                        eta: eta.asRgb(),
                        extinction: extinctionParameter.asRgb(),
                        maxDepth: maxDepth,
                        nSamples: nSamples,
                        remapRoughness: remapRoughness)
        }
}
