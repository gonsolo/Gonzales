import Foundation

final class TransformedPrimitive: Boundable & Intersectable {

        init(accelerator: Accelerator, transform: Transform) {
                self.accelerator = accelerator
                self.transform = transform
        }

        func intersect(
                ray: Ray,
                tHit: inout FloatX,
                material: MaterialIndex,
                interaction: inout SurfaceInteraction
        ) throws {
                let localRay = transform.inverse * ray
                // TODO: transform tHit?
                try accelerator.intersect(
                        ray: localRay,
                        tHit: &tHit,
                        material: material,
                        interaction: &interaction)
                if !interaction.valid {
                        return
                }
                interaction = transform * interaction
        }

        func worldBound() -> Bounds3f {
                let bound = transform * accelerator.worldBound()
                return bound
        }

        func objectBound() -> Bounds3f {
                unimplemented()
        }

        let accelerator: Accelerator
        let transform: Transform
}
