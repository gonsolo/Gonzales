var accelerators = [Accelerator]()

enum Accelerator: Boundable, Intersectable {

        case boundingHierarchy(BoundingHierarchy)
        case embree(EmbreeAccelerator)
        case optix(Optix)

        func intersect(
                ray: Ray,
                tHit: inout FloatX,
                interaction: inout SurfaceInteraction
        ) throws {
                switch self {
                case .boundingHierarchy(let boundingHierarchy):
                        try boundingHierarchy.intersect(
                                ray: ray,
                                tHit: &tHit,
                                interaction: &interaction)
                case .embree(let embree):
                        try embree.intersect(
                                ray: ray,
                                tHit: &tHit,
                                interaction: &interaction)
                case .optix(let optix):
                        try optix.intersect(
                                ray: ray,
                                tHit: &tHit,
                                interaction: &interaction)
                }
        }

        func objectBound() -> Bounds3f {
                switch self {
                case .boundingHierarchy(let boundingHierarchy):
                        return boundingHierarchy.objectBound()
                case .embree(let embree):
                        return embree.objectBound()
                case .optix(let optix):
                        return optix.objectBound()
                }
        }

        func worldBound() -> Bounds3f {
                switch self {
                case .boundingHierarchy(let boundingHierarchy):
                        return boundingHierarchy.worldBound()
                case .embree(let embree):
                        return embree.worldBound()
                case .optix(let optix):
                        return optix.worldBound()
                }
        }

}
