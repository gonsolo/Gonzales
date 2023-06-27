var accelerators = [Accelerator]()

enum Accelerator: Boundable, Intersectable {

        case embree(EmbreeAccelerator)
        case boundingHierarchy(BoundingHierarchy)

        func intersect(
                ray: Ray,
                tHit: inout FloatX,
                material: MaterialIndex,
                interaction: inout SurfaceInteraction
        ) throws {
                switch self {
                case .embree(let embree):
                        try embree.intersect(
                                ray: ray,
                                tHit: &tHit,
                                material: material,
                                interaction: &interaction)
                case .boundingHierarchy(let boundingHierarchy):
                        try boundingHierarchy.intersect(
                                ray: ray,
                                tHit: &tHit,
                                material: material,
                                interaction: &interaction)
                }
        }

        func objectBound() -> Bounds3f {
                switch self {
                case .embree(let embree):
                        return embree.objectBound()
                case .boundingHierarchy(let boundingHierarchy):
                        return boundingHierarchy.objectBound()
                }
        }

        func worldBound() -> Bounds3f {
                switch self {
                case .embree(let embree):
                        return embree.worldBound()
                case .boundingHierarchy(let boundingHierarchy):
                        return boundingHierarchy.worldBound()
                }
        }

}
