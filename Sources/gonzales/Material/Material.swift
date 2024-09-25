///        A material provides the look of a surface.

enum Material {
        case areaLight(AreaLight)
        case coatedDiffuse(CoatedDiffuse)
        case conductor(Conductor)
        case dielectric(Dielectric)
        case diffuse(Diffuse)
        case diffuseTransmission(DiffuseTransmission)
        case geometricPrimitive(GeometricPrimitive)
        case hair(Hair)
        case interface(Interface)
        case measured(Measured)

        @MainActor
        func getBsdf(interaction: Interaction) -> GlobalBsdf {
                switch self {
                case .areaLight(let areaLight):
                        return areaLight.getBsdf(interaction: interaction)
                case .coatedDiffuse(let coatedDiffuse):
                        return coatedDiffuse.getBsdf(interaction: interaction)
                case .conductor(let conductor):
                        return conductor.getBsdf(interaction: interaction)
                case .dielectric(let dielectric):
                        return dielectric.getBsdf(interaction: interaction)
                case .diffuse(let diffuse):
                        return diffuse.getBsdf(interaction: interaction)
                case .diffuseTransmission(let diffuseTransmission):
                        return diffuseTransmission.getBsdf(interaction: interaction)
                case .geometricPrimitive(let geometricPrimitive):
                        return geometricPrimitive.getBsdf(interaction: interaction)
                case .hair(let hair):
                        return hair.getBsdf(interaction: interaction)
                case .interface(let interface):
                        return interface.getBsdf()
                case .measured(let measured):
                        return measured.getBsdf(interaction: interaction)
                }
        }

        var isInterface: Bool {
                switch self {
                case .interface:
                        return true
                default:
                        return false
                }
        }
}
