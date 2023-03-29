var intersectionTests = 0

func intersect(
        ray: Ray,
        tHit: inout FloatX,
        interaction: inout SurfaceInteraction,
        hierarchy: Accelerator
) throws {
        intersectionTests += 1
        try hierarchy.intersect(
                ray: ray,
                tHit: &tHit,
                material: -1,
                interaction: &interaction)
}

struct Scene {

        init(accelerator: Accelerator, lights: [Light]) {
                self.accelerator = accelerator
                self.lights = lights
                infiniteLights = lights.compactMap { $0 as? InfiniteLight }
                sceneDiameter = diameter()
        }

        func bound() -> Bounds3f {
                return accelerator.worldBound()
        }

        func diameter() -> FloatX {
                return length(bound().diagonal())
        }

        static func statistics() {
                print("  Ray (regular + shadow) intersection tests:\t\t\t\t\(intersectionTests)")
        }

        var accelerator: Accelerator
        var lights: [Light]
        var infiniteLights: [Light]
}
