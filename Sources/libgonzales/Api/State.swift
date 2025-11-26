struct UninstancedMaterial {
        let type: String
        let parameters: ParameterDictionary
}

struct ImmutableState {
        let namedMedia: [String: any Medium]
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
                namedMedia = [String: any Medium]()
                textures = [String: Texture]()
                ptexCache = PtexCache()
        }

        @MainActor
        private func makeDefaultMaterial(insteadOf material: String) throws -> Material {
                print("Unknown material \"\(material)\". Creating default.")
                var parameters = ParameterDictionary()
                parameters["reflectance"] = [gray]
                let diffuse = try createDiffuse(parameters: parameters)
                return Material.diffuse(diffuse)
        }

        @MainActor
        func makeMaterial(type: String, parameters: ParameterDictionary) throws -> Material {

                var material: Material
                switch type {
                case "coateddiffuse":
                        let coatedDiffuse = try createCoatedDiffuse(parameters: parameters)
                        material = Material.coatedDiffuse(coatedDiffuse)
                // coatedconductor missing
                case "conductor":
                        let conductor = try createConductor(parameters: parameters)
                        material = Material.conductor(conductor)
                case "dielectric":
                        let dielectric = try createDielectric(parameters: parameters)
                        material = Material.dielectric(dielectric)
                case "diffuse":
                        let diffuse = try createDiffuse(parameters: parameters)
                        material = Material.diffuse(diffuse)
                case "diffusetransmission":
                        let diffuseTransmission = try createDiffuseTransmission(parameters: parameters)
                        material = Material.diffuseTransmission(diffuseTransmission)
                case "hair":
                        let hair = try createHair(parameters: parameters)
                        material = Material.hair(hair)
                case "interface":
                        let interface = try createInterface(parameters: parameters)
                        material = Material.interface(interface)
                case "measured":
                        let measured = try createMeasured(parameters: parameters)
                        material = Material.measured(measured)
                // mix missing
                // subsurface missing
                // thindielectric missing
                default:
                        material = try makeDefaultMaterial(insteadOf: type)
                }
                return material
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
                return try makeMaterial(type: material.type, parameters: merged)
        }

        func getImmutable() -> ImmutableState {
                return ImmutableState(namedMedia: self.namedMedia)
        }

        var areaLight = ""
        var areaLightParameters = ParameterDictionary()

        var currentMediumInterface: MediumInterface?
        var currentNamedMaterial: String
        var currentMaterial: UninstancedMaterial?

        var namedMaterials: [String: UninstancedMaterial]
        var namedMedia: [String: any Medium]

        var objectName: String?

        var textures: [String: Texture]

        let ptexCache: PtexCache
}
