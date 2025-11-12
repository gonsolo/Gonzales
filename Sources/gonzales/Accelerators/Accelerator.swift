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
        ) throws -> Bool {
                try boundingHierarchy.intersect(
                        scene: scene,
                        ray: ray,
                        tHit: &tHit)
        }

        func intersect(
                scene: Scene,
                ray: Ray,
                tHit: inout FloatX,
                interaction: inout SurfaceInteraction
        ) throws {
                try boundingHierarchy.intersect(
                        scene: scene,
                        ray: ray,
                        tHit: &tHit,
                        interaction: &interaction)
        }

        func objectBound(scene: Scene) -> Bounds3f {
                return boundingHierarchy.objectBound(scene: scene)
        }

        func worldBound(scene: Scene) -> Bounds3f {
                return boundingHierarchy.worldBound(scene: scene)
        }
}
