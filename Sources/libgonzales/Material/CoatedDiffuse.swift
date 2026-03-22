struct CoatedDiffuse {

        init(
                roughness: (Real, Real),
                reflectance: RgbSpectrumTexture,
                refractiveIndex: FloatTexture,
                thickness: FloatTexture,
                albedo: RgbSpectrumTexture,
                asymmetry: FloatTexture,
                maxDepth: Int,
                nSamples: Int,
                remapRoughness: Bool
        ) {
                self.roughness = roughness
                self.reflectance = reflectance
                self.refractiveIndex = refractiveIndex
                self.thickness = thickness
                self.albedo = albedo
                self.asymmetry = asymmetry
                self.maxDepth = maxDepth
                self.nSamples = nSamples
                self.remapRoughness = remapRoughness
        }

        func getBsdf(interaction: any Interaction) -> CoatedDiffuseBsdf {
                let refractiveIndex = self.refractiveIndex.evaluateFloat(at: interaction)
                let reflectanceAtInteraction = reflectance.evaluateRgbSpectrum(at: interaction)
                let thicknessAtInteraction = self.thickness.evaluateFloat(at: interaction)
                let albedoAtInteraction = self.albedo.evaluateRgbSpectrum(at: interaction)
                let asymmetryAtInteraction = self.asymmetry.evaluateFloat(at: interaction)
                let bsdfFrame = BsdfFrame(interaction: interaction)

                var alpha: (Real, Real) = roughness
                if remapRoughness {
                        alpha = TrowbridgeReitzDistribution.getAlpha(from: roughness)
                }
                
                // Ensure alpha is at least 0.001 to avoid NaN or division by zero in microfacet math
                alpha.0 = max(alpha.0, 0.001)
                alpha.1 = max(alpha.1, 0.001)
                
                let distribution = TrowbridgeReitzDistribution(alpha: alpha)
                let dielectric = DielectricBsdf(
                        distribution: distribution, refractiveIndex: refractiveIndex, bsdfFrame: bsdfFrame)
                let diffuse = DiffuseBsdf(reflectance: reflectanceAtInteraction, bsdfFrame: bsdfFrame)

                let coatedDiffuseBsdf = CoatedDiffuseBsdf(
                        dielectric: dielectric,
                        diffuse: diffuse,
                        thickness: Real(thicknessAtInteraction),
                        albedo: albedoAtInteraction,
                        asymmetry: Real(asymmetryAtInteraction),
                        maxDepth: maxDepth,
                        nSamples: nSamples,
                        bsdfFrame: bsdfFrame)
                return coatedDiffuseBsdf
        }

        var reflectance: RgbSpectrumTexture
        var refractiveIndex: FloatTexture
        var roughness: (Real, Real)
        var thickness: FloatTexture
        var albedo: RgbSpectrumTexture
        var asymmetry: FloatTexture
        var maxDepth: Int
        var nSamples: Int
        var remapRoughness: Bool
}

extension CoatedDiffuse {
        static func create(parameters: ParameterDictionary, textures: [String: Texture]) throws -> CoatedDiffuse {
        let remapRoughness = try parameters.findOneBool(called: "remaproughness", else: true)
        let roughnessOptional = try parameters.findOneRealOptional(called: "roughness")
        let uRoughness =
                try roughnessOptional ?? parameters.findOneReal(called: "uroughness", else: 0.5)
        let vRoughness =
                try roughnessOptional ?? parameters.findOneReal(called: "vroughness", else: 0.5)
        let roughness = (uRoughness, vRoughness)
        let reflectance = try parameters.findRgbSpectrumTexture(name: "reflectance", textures: textures)
        let refractiveIndex = try parameters.findRealTexture(name: "eta", textures: textures, else: 1.5)
        
        let thickness = try parameters.findRealTexture(name: "thickness", textures: textures, else: 0.01)
        let asymmetry = try parameters.findRealTexture(name: "g", textures: textures, else: 0.0)
        let maxDepth = try parameters.findOneInt(called: "maxdepth", else: 10)
        let nSamples = try parameters.findOneInt(called: "nsamples", else: 1)
        


        // PBRT defaults albedo to 0.0 (black)
        let albedo = try parameters.findRgbSpectrumTexture(name: "albedo", textures: textures, else: RgbSpectrum(intensity: 0.0))

        return CoatedDiffuse(
                roughness: roughness,
                reflectance: reflectance,
                refractiveIndex: refractiveIndex,
                thickness: thickness,
                albedo: albedo,
                asymmetry: asymmetry,
                maxDepth: maxDepth,
                nSamples: nSamples,
                remapRoughness: remapRoughness)
}
}
