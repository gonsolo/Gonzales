///        A light source

enum Light {

        case area(AreaLight)
        case infinite(InfiniteLight)
        case distant(DistantLight)
        case point(PointLight)

        func sample(for reference: Interaction, u: TwoRandomVariables) -> (
                radiance: RgbSpectrum,
                direction: Vector,
                pdf: FloatX,
                visibility: Visibility
        ) {
                switch self {
                case .area(let areaLight):
                        return areaLight.sample(for: reference, u: u)
                case .infinite(let infiniteLight):
                        return infiniteLight.sample(for: reference, u: u)
                case .distant(let distantLight):
                        return distantLight.sample(for: reference, u: u)
                case .point(let pointLight):
                        return pointLight.sample(for: reference, u: u)
                }
        }

        //@_noAllocation
        func probabilityDensityFor(samplingDirection direction: Vector, from reference: Interaction)
                throws -> FloatX
        {
                switch self {
                case .area(let areaLight):
                        return try areaLight.probabilityDensityFor(
                                samplingDirection: direction,
                                from: reference)
                case .infinite(let infiniteLight):
                        return try infiniteLight.probabilityDensityFor(
                                samplingDirection: direction,
                                from: reference)
                case .distant(let distantLight):
                        return try distantLight.probabilityDensityFor(
                                samplingDirection: direction,
                                from: reference)
                case .point(let pointLight):
                        return try pointLight.probabilityDensityFor(
                                samplingDirection: direction,
                                from: reference)
                }
        }

        func radianceFromInfinity(for ray: Ray) -> RgbSpectrum {
                switch self {
                case .area(let areaLight):
                        return areaLight.radianceFromInfinity(for: ray)
                case .infinite(let infiniteLight):
                        return infiniteLight.radianceFromInfinity(for: ray)
                case .distant(let distantLight):
                        return distantLight.radianceFromInfinity(for: ray)
                case .point(let pointLight):
                        return pointLight.radianceFromInfinity(for: ray)
                }
        }

        func power() -> FloatX {
                switch self {
                case .area(let areaLight):
                        return areaLight.power()
                case .infinite(let infiniteLight):
                        return infiniteLight.power()
                case .distant(let distantLight):
                        return distantLight.power()
                case .point(let pointLight):
                        return pointLight.power()
                }
        }

        var isDelta: Bool {
                switch self {
                case .area:
                        return true
                case .infinite:
                        return false
                case .distant:
                        return true
                case .point:
                        return true
                }
        }
}
