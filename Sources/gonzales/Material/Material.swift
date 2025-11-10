///        A material provides the look of a surface.

enum Material {
        case areaLight(AreaLight)
        case coatedDiffuse(CoatedDiffuse)
        case conductor(Conductor)
        case dielectric(Dielectric)
        case diffuse(Diffuse)
        case diffuseTransmission(DiffuseTransmission)
        case hair(Hair)
        case interface(Interface)
        case measured(Measured)

        func getBsdf(interaction: SurfaceInteraction) -> GlobalBsdfType {
                switch self {
                case .areaLight(let areaLight):
                        let bsdf = areaLight.getBsdf(interaction: interaction)
                        return .diffuseBsdf(bsdf)
                case .coatedDiffuse(let coatedDiffuse):
                        let bsdf = coatedDiffuse.getBsdf(interaction: interaction)
                        return .coatedDiffuseBsdf(bsdf)
                case .conductor(let conductor):
                        let bsdf = conductor.getBsdf(interaction: interaction)
                        return .microfacetReflection(bsdf)
                case .dielectric(let dielectric):
                        let bsdf = dielectric.getBsdf(interaction: interaction)
                        return .dielectricBsdf(bsdf)
                case .diffuse(let diffuse):
                        let bsdf = diffuse.getBsdf(interaction: interaction)
                        return .diffuseBsdf(bsdf)
                case .diffuseTransmission(let diffuseTransmission):
                        let bsdf = diffuseTransmission.getBsdf(interaction: interaction)
                        return .diffuseBsdf(bsdf)
                case .hair(let hair):
                        let bsdf = hair.getBsdf(interaction: interaction)
                        return .hairBsdf(bsdf)
                case .interface(let interface):
                        return interface.getBsdf()
                case .measured(let measured):
                        let bsdf = measured.getBsdf(interaction: interaction)
                        return .diffuseBsdf(bsdf)
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
