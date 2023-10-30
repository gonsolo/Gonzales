typealias AcceleratorIndex = Int

struct Scene {

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

        func intersect(
                rays: [Ray],
                tHits: inout [FloatX],
                interactions: inout [SurfaceInteraction],
                skips: [Bool]
        ) throws {
                try accelerators[acceleratorIndex].intersect(
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
                try accelerators[acceleratorIndex].intersect(
                        ray: ray,
                        tHit: &tHit,
                        interaction: &interaction)
        }

        func bound() -> Bounds3f {
                return accelerators[acceleratorIndex].worldBound()
        }

        func diameter() -> FloatX {
                return length(bound().diagonal())
        }

        var acceleratorIndex: AcceleratorIndex
        var lights: [Light]
        var infiniteLights: [InfiniteLight]
}
