struct Scene: Sendable {

        @MainActor
        init(accelerator: Accelerator, lights: [Light]) {
                self.accelerator = accelerator
                self.lights = lights
                infiniteLights = lights.compactMap {
                        switch $0 {
                        case .infinite(let infiniteLight):
                                return infiniteLight
                        default:
                                return nil
                        }
                }
        }

        func intersect(
                rays: Ray,
                tHits: inout FloatX,
                interactions: inout SurfaceInteraction,
                skips: [Bool]
        ) throws {
                try accelerator.intersect(
                        ray: rays,
                        tHit: &tHits,
                        interaction: &interactions)
        }

        func intersect(
                ray: Ray,
                tHit: inout FloatX
        ) throws -> Bool {
                return try accelerator.intersect(
                        ray: ray,
                        tHit: &tHit)
        }

        func intersect(
                ray: Ray,
                tHit: inout FloatX,
                interaction: inout SurfaceInteraction,
                skip: Bool = false
        ) throws {
                if skip {
                        return
                }
                try accelerator.intersect(
                        ray: ray,
                        tHit: &tHit,
                        interaction: &interaction)
        }

        @MainActor
        func bound() -> Bounds3f {
                return accelerator.worldBound()
        }

        @MainActor
        func diameter() -> FloatX {
                return length(bound().diagonal())
        }

        var accelerator: Accelerator
        var lights: [Light]
        var infiniteLights: [InfiniteLight]
}
