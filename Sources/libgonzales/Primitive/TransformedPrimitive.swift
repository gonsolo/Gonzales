import Foundation

struct TransformedPrimitive: Boundable, Intersectable {

        func intersect(
                scene: Scene,
                ray: Ray,
                tHit: inout Real
        ) throws -> SurfaceInteraction? {
                let localRay = transform.inverse * ray
                var interaction = try accelerator.intersect(
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
        ) throws -> Bool {
                let localRay = transform.inverse * worldRay
                var localTHit = tHit
                if try accelerator.intersect(scene: scene, ray: localRay, tHit: &localTHit) != nil {
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
        ) throws -> SurfaceInteraction? {
                let localRay = transform.inverse * worldRay
                var localTHit = data.tValue + 1e-5
                var interaction = try accelerator.intersect(scene: scene, ray: localRay, tHit: &localTHit)
                if let inter = interaction {
                        interaction = transform * inter
                }
                return interaction
        }

        func worldBound(scene: Scene) -> Bounds3f {
                let bound = transform * accelerator.worldBound(scene: scene)
                return bound
        }

        func objectBound(scene _: Scene) throws -> Bounds3f {
                throw RenderError.unimplemented(function: #function, file: #filePath, line: #line, message: "")
        }

        // let acceleratorIndex: AcceleratorIndex
        let accelerator: Accelerator
        let transform: Transform
        let idx: Int
}
