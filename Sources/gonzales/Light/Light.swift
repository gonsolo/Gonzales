///        A light source

enum Light: Sendable {

        case area(AreaLight)
        case infinite(InfiniteLight)
        case distant(DistantLight)
        case point(PointLight)

        func sample(point: Point, u: TwoRandomVariables, accelerator: Accelerator, scene: Scene) -> (
                radiance: RgbSpectrum,
                direction: Vector,
                pdf: FloatX,
                visibility: Visibility
        ) {
                switch self {
                case .area(let areaLight):
                        return areaLight.sample(point: point, u: u, accelerator: accelerator, scene: scene)
                case .infinite(let infiniteLight):
                        return infiniteLight.sample(point: point, u: u, accelerator: accelerator)
                case .distant(let distantLight):
                        return distantLight.sample(point: point, u: u, accelerator: accelerator)
                case .point(let pointLight):
                        return pointLight.sample(point: point, u: u, accelerator: accelerator)
                }
        }

        func probabilityDensityFor<I: Interaction>(scene: Scene, samplingDirection direction: Vector, from reference: I)
                throws -> FloatX
        {
                switch self {
                case .area(let areaLight):
                        return try areaLight.probabilityDensityFor(
                                scene: scene,
                                samplingDirection: direction,
                                from: reference)
                case .infinite(let infiniteLight):
                        return try infiniteLight.probabilityDensityFor(
                                scene: scene,
                                samplingDirection: direction,
                                from: reference)
                case .distant(let distantLight):
                        return try distantLight.probabilityDensityFor(
                                scene: scene,
                                samplingDirection: direction,
                                from: reference)
                case .point(let pointLight):
                        return try pointLight.probabilityDensityFor(
                                scene: scene,
                                samplingDirection: direction,
                                from: reference)
                }
        }

        @MainActor
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

        func power(scene: Scene) -> FloatX {
                switch self {
                case .area(let areaLight):
                        return areaLight.power(scene: scene)
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
