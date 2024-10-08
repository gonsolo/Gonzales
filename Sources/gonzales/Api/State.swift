struct UninstancedMaterial {
        let type: String
        let parameters: ParameterDictionary
}

struct ImmutableState {
        let namedMedia: [String: Medium]
}

struct State {

        @MainActor
        init() {
                namedMaterials = [String: UninstancedMaterial]()
                currentNamedMaterial = "None"
                var parameters = ParameterDictionary()
                parameters["reflectance"] = [gray]
                namedMaterials["None"] = UninstancedMaterial(
                        type: "diffuse",
                        parameters: parameters
                )
                namedMedia = [String: Medium]()
                textures = [String: Texture]()
                ptexCache = PtexCache()
        }

        @MainActor
        func createMaterial(parameters: ParameterDictionary) throws -> Material {
                var material: UninstancedMaterial!
                if currentMaterial != nil {
                        material = currentMaterial!
                } else {
                        assert(currentNamedMaterial != "")
                        guard let named = namedMaterials[currentNamedMaterial] else {
                                warning("The material \(currentNamedMaterial) was not defined!")
                                let diffuse = Diffuse(
                                        reflectance: Texture.rgbSpectrumTexture(
                                                RgbSpectrumTexture.constantTexture(
                                                        ConstantTexture(value: gray))))
                                return Material.diffuse(diffuse)
                        }
                        material = named
                }
                var merged = parameters
                merged.merge(material.parameters) { (current, _) in current }
                return try api.makeMaterial(type: material.type, parameters: merged)
        }

        func getImmutable() -> ImmutableState {
                return ImmutableState(namedMedia: self.namedMedia)
        }

        var areaLight = ""
        var areaLightParameters = ParameterDictionary()

        var currentMediumInterface: MediumInterface? = nil
        var currentNamedMaterial: String
        var currentMaterial: UninstancedMaterial?

        var namedMaterials: [String: UninstancedMaterial]
        var namedMedia: [String: Medium]

        var objectName: String? = nil

        var textures: [String: Texture]

        let ptexCache: PtexCache
}
