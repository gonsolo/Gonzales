import Foundation

final class TransformedPrimitive: Boundable, Intersectable, Sendable {

        func intersect(
                scene: Scene,
                ray: Ray,
                tHit: inout Real
        ) -> SurfaceInteraction? {
                let localRay = transform.inverse * ray
                var interaction = accelerator.intersect(
                        scene: scene,
                        ray: localRay,
                        tHit: &tHit)
                if interaction == nil {
                        return nil
                }
                if let inter = interaction {
                        interaction = transform * inter
                }
                return interaction
        }

        func getIntersectionData(
                scene: Scene,
                ray worldRay: Ray,
                tHit: inout Real,
                data: inout TriangleIntersection
        ) -> Bool {
                let localRay = transform.inverse * worldRay
                var localTHit = tHit
                if accelerator.intersect(scene: scene, ray: localRay, tHit: &localTHit) != nil {
                        tHit = localTHit
                        data = TriangleIntersection(
                                primId: PrimId(id1: idx, id2: -1, type: .transformedPrimitive),
                                tValue: tHit,
                                barycentric0: 0,
                                barycentric1: 0,
                                barycentric2: 0)
                        return true
                }
                return false
        }

        func computeSurfaceInteraction(
                scene: Scene,
                data: TriangleIntersection,
                worldRay: Ray
        ) -> SurfaceInteraction? {
                let localRay = transform.inverse * worldRay
                var localTHit = data.tValue + 1e-5
                var interaction = accelerator.intersect(scene: scene, ray: localRay, tHit: &localTHit)
                if let inter = interaction {
                        interaction = transform * inter
                }
                return interaction
        }

        func worldBound(scene: Scene) -> Bounds3f {
                let bound = transform * accelerator.worldBound(scene: scene)
                return bound
        }

        func objectBound(scene: Scene) -> Bounds3f {
                return accelerator.worldBound(scene: scene)
        }

        let accelerator: Accelerator
        let transform: Transform
        let idx: Int

        init(accelerator: Accelerator, transform: Transform, idx: Int) {
                self.accelerator = accelerator
                self.transform = transform
                self.idx = idx
        }
}
