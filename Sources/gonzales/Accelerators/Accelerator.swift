final class Accelerator: Boundable, Intersectable, Sendable {

        init(boundingHierarchy: BoundingHierarchy) {
                self.boundingHierarchy = boundingHierarchy
        }

        nonisolated(unsafe) private var boundingHierarchy: BoundingHierarchy
        //case boundingHierarchy(BoundingHierarchy)
        //case embree(EmbreeAccelerator)
        //case optix(Optix)

        func intersect(
                rays: [Ray],
                tHits: inout [FloatX],
                interactions: inout [SurfaceInteraction],
                skips: [Bool]
        ) throws {
                //switch self {
                //case .boundingHierarchy(let boundingHierarchy):
                        for i in 0..<rays.count {
                                if !skips[i] {
                                        try boundingHierarchy.intersect(
                                                ray: rays[i],
                                                tHit: &tHits[i],
                                                interaction: &interactions[i])
                                }
                        }
                //case .embree(let embree):
                //        for i in 0..<rays.count {
                //                if !skips[i] {
                //                        try embree.intersect(
                //                                ray: rays[i],
                //                                tHit: &tHits[i],
                //                                interaction: &interactions[i])
                //                }
                //        }
                //case .optix(let optix):
                //        try optix.intersect(
                //                rays: rays,
                //                tHits: &tHits,
                //                interactions: &interactions,
                //                skips: skips)
                }
        //}

        func intersect(
                ray: Ray,
                tHit: inout FloatX
        ) throws -> Bool {
                //switch self {
                //case .boundingHierarchy(let boundingHierarchy):
                        try boundingHierarchy.intersect(
                                ray: ray,
                                tHit: &tHit)
                //}
        }

        func intersect(
                ray: Ray,
                tHit: inout FloatX,
                interaction: inout SurfaceInteraction
        ) throws {
                //switch self {
                //case .boundingHierarchy(let boundingHierarchy):
                        try boundingHierarchy.intersect(
                                ray: ray,
                                tHit: &tHit,
                                interaction: &interaction)
                //case .embree(let embree):
                //        try embree.intersect(
                //                ray: ray,
                //                tHit: &tHit,
                //                interaction: &interaction)
                //case .optix(let optix):
                //        try optix.intersect(
                //                ray: ray,
                //                tHit: &tHit,
                //                interaction: &interaction)
                //}
        }

        @MainActor
        func objectBound() -> Bounds3f {
                //switch self {
                //case .boundingHierarchy(let boundingHierarchy):
                        return boundingHierarchy.objectBound()
                //case .embree(let embree):
                //        return embree.objectBound()
                //case .optix(let optix):
                //        return optix.objectBound()
                //}
        }

        @MainActor
        func worldBound() -> Bounds3f {
                //switch self {
                //case .boundingHierarchy(let boundingHierarchy):
                        return boundingHierarchy.worldBound()
                //case .embree(let embree):
                //        return embree.worldBound()
                //case .optix(let optix):
                //        return optix.worldBound()
                //}
        }

        func addScene(scene: Scene) {
                boundingHierarchy.addScene(scene: scene)
        }
}
