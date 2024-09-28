import Foundation

final class TransformedPrimitive: @preconcurrency Boundable, Intersectable, Sendable {

        init(acceleratorIndex: AcceleratorIndex, transform: Transform) {
                self.acceleratorIndex = acceleratorIndex
                self.transform = transform
        }

        @MainActor
        func intersect(
                ray: Ray,
                tHit: inout FloatX,
                interaction: inout SurfaceInteraction
        ) async throws {
                let localRay = transform.inverse * ray
                // TODO: transform tHit?
                try await accelerators[acceleratorIndex].intersect(
                        ray: localRay,
                        tHit: &tHit,
                        interaction: &interaction)
                if !interaction.valid {
                        return
                }
                interaction = transform * interaction
        }

        @MainActor
        func worldBound() -> Bounds3f {
                let bound = transform * accelerators[acceleratorIndex].worldBound()
                return bound
        }

        func objectBound() -> Bounds3f {
                unimplemented()
        }

        let acceleratorIndex: AcceleratorIndex
        let transform: Transform
}
