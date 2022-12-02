import Foundation

final class TransformedPrimitive: Boundable & Intersectable {

        init(primitive: Boundable & Intersectable, transform: Transform) {
                self.primitive = primitive
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
                try primitive.intersect(
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
                let bound = transform * primitive.worldBound()
                return bound
        }

        func objectBound() -> Bounds3f {
                unimplemented()
        }

        let primitive: Boundable & Intersectable
        let transform: Transform
}
