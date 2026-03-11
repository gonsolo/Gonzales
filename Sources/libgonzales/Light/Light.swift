///        A light source

protocol LightSource: Sendable {
        func sample(point: Point, samples: TwoRandomVariables, accelerator: Accelerator, scene: Scene)
                -> LightSample
        func probabilityDensityFor<I: Interaction>(
                scene: Scene, samplingDirection direction: Vector, from reference: I
        )
                throws -> FloatX
        func radianceFromInfinity(for ray: Ray) -> RgbSpectrum
        func power(scene: Scene) -> FloatX
        var isDelta: Bool { get }
}

enum Light: Sendable {

        case area(AreaLight)
        case infinite(InfiniteLight)
        case distant(DistantLight)
        case point(PointLight)

        var source: any LightSource {
                switch self {
                case .area(let l): return l
                case .infinite(let l): return l
                case .distant(let l): return l
                case .point(let l): return l
                }
        }

        func sample(point: Point, samples: TwoRandomVariables, accelerator: Accelerator, scene: Scene)
                -> LightSample {
                return source.sample(point: point, samples: samples, accelerator: accelerator, scene: scene)
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

        func power(scene: Scene) -> FloatX {
                return source.power(scene: scene)
        }

        var isDelta: Bool {
                return source.isDelta
        }
}
