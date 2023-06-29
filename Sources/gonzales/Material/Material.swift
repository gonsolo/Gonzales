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

        func setBsdf(interaction: inout SurfaceInteraction) {
                switch self {
                case .areaLight(let areaLight):
                        areaLight.setBsdf(interaction: &interaction)
                case .coatedDiffuse(let coatedDiffuse):
                        coatedDiffuse.setBsdf(interaction: &interaction)
                case .conductor(let conductor):
                        conductor.setBsdf(interaction: &interaction)
                case .dielectric(let dielectric):
                        dielectric.setBsdf(interaction: &interaction)
                case .diffuse(let diffuse):
                        diffuse.setBsdf(interaction: &interaction)
                case .diffuseTransmission(let diffuseTransmission):
                        diffuseTransmission.setBsdf(interaction: &interaction)
                case .geometricPrimitive(let geometricPrimitive):
                        geometricPrimitive.setBsdf(interaction: &interaction)
                case .hair(let hair):
                        hair.setBsdf(interaction: &interaction)
                case .interface(let interface):
                        interface.setBsdf(interaction: &interaction)
                case .measured(let measured):
                        measured.setBsdf(interaction: &interaction)
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
