struct Scene {

        var intersectionTests = 0

        init() {
                primitive = nil
                lights = []
                infiniteLights = []
        }

        init(aggregate: Boundable & Intersectable, lights: [Light]) {
                self.primitive = aggregate
                self.lights = lights
                infiniteLights = lights.compactMap { $0 as? InfiniteLight }
        }

        mutating func intersect(ray: Ray, tHit: inout FloatX) throws -> SurfaceInteraction {
                intersectionTests += 1
                guard let primitive else {
                        return SurfaceInteraction()
                }
                let interaction = try primitive.intersect(ray: ray, tHit: &tHit, material: -1)
                return interaction
        }

        func bound() -> Bounds3f {
                guard let p = primitive else {
                        return Bounds3f()
                }
                return p.worldBound()
        }

        func diameter() -> FloatX {
                return length(bound().diagonal())
        }

        func statistics() {
                print("  Ray (regular + shadow) intersection tests:\t\t\t\t\(intersectionTests)")
        }

        var primitive: (Boundable & Intersectable)?
        var lights: [Light]
        var infiniteLights: [Light]
}
