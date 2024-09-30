typealias AcceleratorIndex = Int

struct Scene: Sendable {

        @MainActor
        init(acceleratorIndex: AcceleratorIndex, lights: [Light]) {
                self.acceleratorIndex = acceleratorIndex
                self.lights = lights
                infiniteLights = lights.compactMap {
                        switch $0 {
                        case .infinite(let infiniteLight):
                                return infiniteLight
                        default:
                                return nil
                        }
                }
                sceneDiameter = diameter()
        }

        @MainActor
        func intersect(
                rays: [Ray],
                tHits: inout [FloatX],
                interactions: inout [SurfaceInteraction],
                skips: [Bool]
        ) async throws {
                try await accelerators[acceleratorIndex].intersect(
                        rays: rays,
                        tHits: &tHits,
                        interactions: &interactions,
                        skips: skips)
        }

        @MainActor
        func intersect(
                ray: Ray,
                tHit: inout FloatX,
                interaction: inout SurfaceInteraction,
                skip: Bool = false
        ) async throws {
                if skip {
                        return
                }
                try await accelerators[acceleratorIndex].intersect(
                        ray: ray,
                        tHit: &tHit,
                        interaction: &interaction)
        }

        @MainActor
        func bound() -> Bounds3f {
                return accelerators[acceleratorIndex].worldBound()
        }

        @MainActor
        func diameter() -> FloatX {
                return length(bound().diagonal())
        }

        var acceleratorIndex: AcceleratorIndex
        var lights: [Light]
        var infiniteLights: [InfiniteLight]
}
