import Foundation

final class TransformedPrimitive: Boundable & Intersectable {

        init(acceleratorIndex: AcceleratorIndex, transform: Transform) {
                self.acceleratorIndex = acceleratorIndex
                self.transform = transform
        }

        func intersect(
                ray: Ray,
                tHit: inout FloatX,
                interaction: inout SurfaceInteraction
        ) throws {
                let localRay = transform.inverse * ray
                // TODO: transform tHit?
                try accelerators[acceleratorIndex].intersect(
                        ray: localRay,
                        tHit: &tHit,
                        interaction: &interaction)
                if !interaction.valid {
                        return
                }
                interaction = transform * interaction
        }

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
