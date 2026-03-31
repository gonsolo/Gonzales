struct UninstancedMaterial {
        let type: String
        let parameters: ParameterDictionary
}

struct ImmutableState {
        let namedMedia: [String: any Medium]
}

struct State {

        init(ptexMemory: Int) {
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
                arena = TextureArena()
                ptexCache = PtexCache(ptexMemory: ptexMemory)
        }

        mutating private func makeDefaultMaterial(insteadOf material: String) throws -> Material {
                print("Unknown material \"\(material)\". Creating default.")
                var parameters = ParameterDictionary()
                parameters["reflectance"] = [gray]
                let diffuse = try Diffuse.create(parameters: parameters, textures: textures, arena: &arena)
                return Material.diffuse(diffuse)
        }

        mutating func makeMaterial(
                type: String, parameters: ParameterDictionary, textures: [String: Texture]
        )
                throws -> Material
        {

                var material: Material
                switch type {
                case "coateddiffuse":
                        let coatedDiffuse = try CoatedDiffuse.create(
                                parameters: parameters, textures: textures, arena: &arena)
                        material = Material.coatedDiffuse(coatedDiffuse)
                case "coatedconductor":
                        let coatedConductor = try CoatedConductor.create(
                                parameters: parameters, textures: textures, arena: &arena)
                        material = Material.coatedConductor(coatedConductor)
                case "conductor":
                        let conductor = try Conductor.create(parameters: parameters, arena: &arena)
                        material = Material.conductor(conductor)
                case "dielectric":
                        let dielectric = try Dielectric.create(
                                parameters: parameters, textures: textures, arena: &arena)
                        material = Material.dielectric(dielectric)
                case "diffuse":
                        let diffuse = try Diffuse.create(
                                parameters: parameters, textures: textures, arena: &arena)
                        material = Material.diffuse(diffuse)
                case "diffusetransmission":
                        let diffuseTransmission = try DiffuseTransmission.create(
                                parameters: parameters, textures: textures, arena: &arena)
                        material = Material.diffuseTransmission(diffuseTransmission)
                case "hair":
                        let hair = try Hair.create(parameters: parameters, textures: textures, arena: &arena)
                        material = Material.hair(hair)
                case "interface":
                        let interface = try Interface.create(parameters: parameters)
                        material = Material.interface(interface)
                case "measured":
                        let measured = try Measured.create(parameters: parameters)
                        material = Material.measured(measured)
                case "mix":
                        let mix = try MixMaterial.create(
                                parameters: parameters, textures: textures,
                                namedMaterials: self.namedMaterials, state: &self)
                        material = Material.mix(mix)
                // subsurface missing
                // thindielectric missing
                default:
                        material = try makeDefaultMaterial(insteadOf: type)
                }
                return material
        }

        mutating func createMaterial(
                parameters: ParameterDictionary,
                currentMaterial: UninstancedMaterial?,
                currentNamedMaterial: String,
                textures: [String: Texture]? = nil
        ) throws -> Material {
                var material: UninstancedMaterial!
                if let paramMaterial = currentMaterial {
                        material = paramMaterial
                } else {
                        assert(currentNamedMaterial != "")
                        guard let named = namedMaterials[currentNamedMaterial] else {
                                print("Warning: The material \(currentNamedMaterial) was not defined!")
                                let idx = arena.appendRgb(
                                        RgbSpectrumTexture.constantTexture(ConstantTexture(value: gray)))
                                let diffuse = Diffuse(reflectance: Texture.rgbSpectrumTexture(idx))
                                return Material.diffuse(diffuse)
                        }
                        material = named
                }
                var merged = parameters
                merged.merge(material.parameters) { (current, _) in current }
                return try makeMaterial(
                        type: material.type, parameters: merged, textures: textures ?? self.textures)
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
        var reverseOrientation = false

        let ptexCache: PtexCache
        var arena: TextureArena
}
