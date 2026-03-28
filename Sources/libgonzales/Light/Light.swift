///        A light source

protocol LightSource: Sendable {
        func sample(point: Point, samples: TwoRandomVariables, accelerator: Accelerator, scene: Scene)
                -> LightSample
        func probabilityDensityFor<I: Interaction>(
                scene: Scene, samplingDirection direction: Vector, from reference: I
        )
                -> Real
        func radianceFromInfinity(for ray: Ray, arena: TextureArena) -> RgbSpectrum
        func power(scene: Scene) -> Real
        var isDelta: Bool { get }
}

enum Light: Sendable {

        case area(AreaLight)
        case infinite(InfiniteLight)
        case distant(DistantLight)
        case point(PointLight)

        func sample(point: Point, samples: TwoRandomVariables, accelerator: Accelerator, scene: Scene)
                -> LightSample
        {
                switch self {
                case .area(let light):
                        return light.sample(
                                point: point, samples: samples, accelerator: accelerator, scene: scene)
                case .infinite(let light):
                        return light.sample(
                                point: point, samples: samples, accelerator: accelerator, scene: scene)
                case .distant(let light):
                        return light.sample(
                                point: point, samples: samples, accelerator: accelerator, scene: scene)
                case .point(let light):
                        return light.sample(
                                point: point, samples: samples, accelerator: accelerator, scene: scene)
                }
        }

        func probabilityDensityFor<I: Interaction>(
                scene: Scene, samplingDirection direction: Vector, from reference: I
        )
                -> Real
        {
                switch self {
                case .area(let light):
                        return light.probabilityDensityFor(
                                scene: scene, samplingDirection: direction, from: reference)
                case .infinite(let light):
                        return light.probabilityDensityFor(
                                scene: scene, samplingDirection: direction, from: reference)
                case .distant(let light):
                        return light.probabilityDensityFor(
                                scene: scene, samplingDirection: direction, from: reference)
                case .point(let light):
                        return light.probabilityDensityFor(
                                scene: scene, samplingDirection: direction, from: reference)
                }
        }

        func radianceFromInfinity(for ray: Ray, arena: TextureArena) -> RgbSpectrum {
                switch self {
                case .area(let light): return light.radianceFromInfinity(for: ray, arena: arena)
                case .infinite(let light): return light.radianceFromInfinity(for: ray, arena: arena)
                case .distant(let light): return light.radianceFromInfinity(for: ray, arena: arena)
                case .point(let light): return light.radianceFromInfinity(for: ray, arena: arena)
                }
        }

        func power(scene: Scene) -> Real {
                switch self {
                case .area(let light): return light.power(scene: scene)
                case .infinite(let light): return light.power(scene: scene)
                case .distant(let light): return light.power(scene: scene)
                case .point(let light): return light.power(scene: scene)
                }
        }

        var isDelta: Bool {
                switch self {
                case .area(let light): return light.isDelta
                case .infinite(let light): return light.isDelta
                case .distant(let light): return light.isDelta
                case .point(let light): return light.isDelta
                }
        }
}
