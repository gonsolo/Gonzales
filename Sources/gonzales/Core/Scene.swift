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
                rays: [Ray],
                tHits: inout [FloatX],
                interactions: inout [SurfaceInteraction],
                skips: [Bool]
        ) throws {
                try accelerator.intersect(
                        rays: rays,
                        tHits: &tHits,
                        interactions: &interactions,
                        skips: skips)
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
                let candidate = try accelerator.intersect_lean(
                        ray: ray,
                        tHit: &tHit)
                let myInteraction = try candidate?.computeInteraction(
                        ray: ray,
                        tHit: &tHit)
                if let myInteraction {
                        interaction = myInteraction
                }
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
