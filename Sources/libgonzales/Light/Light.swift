///        A light source

protocol LightSource: Sendable {
        func sample(point: Point, samples: TwoRandomVariables, accelerator: Accelerator, scene: Scene)
                throws -> LightSample
        func probabilityDensityFor<I: Interaction>(
                scene: Scene, samplingDirection direction: Vector, from reference: I
        )
                throws -> Real
        func radianceFromInfinity(for ray: Ray, arena: TextureArena) -> RgbSpectrum
        func power(scene: Scene) throws -> Real
        var isDelta: Bool { get }
}

enum Light: Sendable {

        case area(AreaLight)
        case infinite(InfiniteLight)
        case distant(DistantLight)
        case point(PointLight)
}
