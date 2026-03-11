import Foundation

struct TransformedPrimitive: Boundable, Intersectable {

        func intersect(
                scene: Scene,
                ray: Ray,
                tHit: inout FloatX
        ) throws -> SurfaceInteraction? {
                let localRay = transform.inverse * ray
                var interaction = try accelerator.intersect(
                        scene: scene,
                        ray: localRay,
                        tHit: &tHit)
                if interaction?.valid == false {
                        return nil
                }
                if let inter = interaction {
                        interaction = transform * inter
                }
                return interaction
        }

        func worldBound(scene: Scene) -> Bounds3f {
                let bound = transform * accelerator.worldBound(scene: scene)
                return bound
        }

        func objectBound(scene _: Scene) -> Bounds3f {
                unimplemented()
        }

        // let acceleratorIndex: AcceleratorIndex
        let accelerator: Accelerator
        let transform: Transform
        let idx: Int
}
