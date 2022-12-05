var intersectionTests = 0

struct Scene {

        init(aggregate: BoundingHierarchy, lights: [Light]) {
                self.primitive = aggregate
                self.lights = lights
                infiniteLights = lights.compactMap { $0 as? InfiniteLight }
                sceneDiameter = diameter()
        }

        func intersect(
                ray: Ray,
                tHit: inout FloatX,
                interaction: inout SurfaceInteraction
        ) throws {
                intersectionTests += 1
                try primitive.intersect(
                        ray: ray,
                        tHit: &tHit,
                        material: -1,
                        interaction: &interaction)
        }

        func bound() -> Bounds3f {
                return primitive.worldBound()
        }

        func diameter() -> FloatX {
                return length(bound().diagonal())
        }

        static func statistics() {
                print("  Ray (regular + shadow) intersection tests:\t\t\t\t\(intersectionTests)")
        }

        var primitive: BoundingHierarchy
        var lights: [Light]
        var infiniteLights: [Light]
}
