struct Scene {

        var intersectionTests = 0

        init() {
                primitive = nil
                lights = []
                infiniteLights = []
        }

        init(aggregate: BoundingHierarchy, lights: [Light]) {
                self.primitive = aggregate
                self.lights = lights
                infiniteLights = lights.compactMap { $0 as? InfiniteLight }
        }

        //@_noAllocation
        @_semantics("optremark")
        mutating func intersect(
                ray: Ray,
                tHit: inout FloatX,
                interaction: inout SurfaceInteraction
        ) throws {
                intersectionTests += 1
                guard let primitive else {
                        return
                }
                try primitive.intersect(
                        ray: ray,
                        tHit: &tHit,
                        material: -1,
                        interaction: &interaction)
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

        var primitive: BoundingHierarchy?
        var lights: [Light]
        var infiniteLights: [Light]
}
