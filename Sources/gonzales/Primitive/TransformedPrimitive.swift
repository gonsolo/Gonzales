import Foundation

struct TransformedPrimitive: Boundable, Intersectable {

        func intersect(
                scene: Scene,
                ray: Ray,
                tHit: inout FloatX
        ) throws -> Bool {
                let localRay = transform.inverse * ray
                return try accelerator.intersect(
                        scene: scene,
                        ray: localRay,
                        tHit: &tHit)
        }

        func intersect(
                scene: Scene,
                ray: Ray,
                tHit: inout FloatX,
                interaction: inout SurfaceInteraction
        ) throws {
                let localRay = transform.inverse * ray
                try accelerator.intersect(
                        scene: scene,
                        ray: localRay,
                        tHit: &tHit,
                        interaction: &interaction)
                if !interaction.valid {
                        return
                }
                interaction = transform * interaction
        }

        @MainActor
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
