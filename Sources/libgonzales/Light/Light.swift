///        A light source

protocol LightSource: Sendable {
        func sample(point: Point, samples: TwoRandomVariables, accelerator: Accelerator, scene: Scene)
                throws -> LightSample
        func probabilityDensityFor<I: Interaction>(
                scene: Scene, samplingDirection direction: Vector, from reference: I
        )
                throws -> FloatX
        func radianceFromInfinity(for ray: Ray) -> RgbSpectrum
        func power(scene: Scene) throws -> FloatX
        var isDelta: Bool { get }
}

enum Light: Sendable {

        case area(AreaLight)
        case infinite(InfiniteLight)
        case distant(DistantLight)
        case point(PointLight)

        var source: any LightSource {
                switch self {
                case .area(let light): return light
                case .infinite(let light): return light
                case .distant(let light): return light
                case .point(let light): return light
                }
        }

        func sample(point: Point, samples: TwoRandomVariables, accelerator: Accelerator, scene: Scene)
                throws -> LightSample {
                return try source.sample(point: point, samples: samples, accelerator: accelerator, scene: scene)
        }

        func probabilityDensityFor<I: Interaction>(
                scene: Scene, samplingDirection direction: Vector, from reference: I
        )
                throws -> FloatX {
                return try source.probabilityDensityFor(
                        scene: scene, samplingDirection: direction, from: reference)
        }

        func radianceFromInfinity(for ray: Ray) -> RgbSpectrum {
                return source.radianceFromInfinity(for: ray)
        }

        func power(scene: Scene) throws -> FloatX {
                return try source.power(scene: scene)
        }

        var isDelta: Bool {
                return source.isDelta
        }
}
