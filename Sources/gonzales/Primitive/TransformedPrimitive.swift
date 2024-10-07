import Foundation

final class TransformedPrimitive: Boundable, Intersectable, Sendable {

        init(
                //acceleratorIndex: AcceleratorIndex,
                accelerator: Accelerator,
                transform: Transform)
        {
                //self.acceleratorIndex = acceleratorIndex
                self.accelerator = accelerator
                self.transform = transform
        }

        func intersect(
                ray: Ray,
                tHit: inout FloatX,
                interaction: inout SurfaceInteraction
        ) throws {
                let localRay = transform.inverse * ray
                // TODO: transform tHit?
                try accelerator.intersect(
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
                let bound = transform * accelerator.worldBound()
                return bound
        }

        func objectBound() -> Bounds3f {
                unimplemented()
        }

        //let acceleratorIndex: AcceleratorIndex
        let accelerator: Accelerator
        let transform: Transform
}
