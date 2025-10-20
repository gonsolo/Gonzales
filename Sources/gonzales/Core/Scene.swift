struct Scene: Sendable {

        @MainActor
        init(lights: [Light], materials: [Material]) {
                self.lights = lights
                self.materials = materials
                self.infiniteLights = lights.compactMap {
                        switch $0 {
                        case .infinite(let infiniteLight):
                                return infiniteLight
                        default:
                                return nil
                        }
                }
                self.immutableTriangleMeshes = triangleMeshBuilder.getMeshes()
                globalScene = self
        }

        mutating func addAccelerator(accelerator: Accelerator) {
                self.accelerator = accelerator
        }

        func intersect(
                ray: Ray,
                tHit: inout FloatX,
                interaction: inout SurfaceInteraction,
                skips: [Bool]
        ) throws {
                try accelerator.intersect(
                        ray: ray,
                        tHit: &tHit,
                        interaction: &interaction)
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

        var accelerator: Accelerator!
        var lights: [Light]
        var infiniteLights: [InfiniteLight]
        let materials: [Material]
        let immutableTriangleMeshes: TriangleMeshes
}

nonisolated(unsafe) var globalScene: Scene?


