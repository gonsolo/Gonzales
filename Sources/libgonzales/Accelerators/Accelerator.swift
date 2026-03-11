struct Accelerator: Boundable, Intersectable, Sendable {

        init() {
                self.boundingHierarchy = BoundingHierarchy(primitives: [], nodes: [])
        }

        init(boundingHierarchy: BoundingHierarchy) {
                self.boundingHierarchy = boundingHierarchy
        }

        private var boundingHierarchy: BoundingHierarchy

        func intersect(
                scene: Scene,
                ray: Ray,
                tHit: inout FloatX
        ) throws -> SurfaceInteraction? {
                return try boundingHierarchy.intersect(
                        scene: scene,
                        ray: ray,
                        tHit: &tHit)
        }

        func objectBound(scene: Scene) -> Bounds3f {
                return boundingHierarchy.objectBound(scene: scene)
        }

        func worldBound(scene: Scene) -> Bounds3f {
                return boundingHierarchy.worldBound(scene: scene)
        }
}
