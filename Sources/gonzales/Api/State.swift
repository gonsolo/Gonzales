struct UninstancedMaterial {
        let type: String
        let parameters: ParameterDictionary
}

struct State {

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

        func createMaterial(parameters: ParameterDictionary) throws -> Material {
                var material: UninstancedMaterial!
                if currentMaterial != nil {
                        material = currentMaterial!
                } else {
                        assert(currentNamedMaterial != "")
                        guard let named = namedMaterials[currentNamedMaterial] else {
                                warning("The material \(currentNamedMaterial) was not defined!")
                                return Diffuse(reflectance: ConstantTexture(value: gray))
                        }
                        material = named
                }
                var merged = parameters
                merged.merge(material.parameters) { (current, _) in current }
                return try api.makeMaterial(type: material.type, parameters: merged)
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
