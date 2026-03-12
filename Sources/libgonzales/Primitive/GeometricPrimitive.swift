struct GeometricPrimitive: Boundable, Intersectable {

        func getIntersectionData(
                scene: Scene,
                ray worldRay: Ray,
                tHit: inout Real,
                data: inout TriangleIntersection
        ) throws -> Bool {
                if alpha == 0 { return false }
                return try shape.getIntersectionData(
                        scene: scene,
                        ray: worldRay,
                        tHit: &tHit,
                        data: &data)
        }

        func computeSurfaceInteraction(
                scene: Scene,
                data: TriangleIntersection,
                worldRay: Ray
        ) throws -> SurfaceInteraction? {
                if alpha == 0 { return nil }
                var interaction = try shape.computeSurfaceInteraction(
                        scene: scene,
                        data: data,
                        worldRay: worldRay)
                if interaction != nil {
                        interaction?.materialIndex = materialIndex
                }
                return interaction
        }

        func intersect(
                scene: Scene,
                ray: Ray,
                tHit: inout Real
        ) throws -> SurfaceInteraction? {
                if alpha == 0 { return nil }
                var interaction = try shape.intersect(
                        scene: scene,
                        ray: ray,
                        tHit: &tHit)
                if interaction != nil {
                        interaction?.materialIndex = materialIndex
                }
                return interaction
        }

        func worldBound(scene: Scene) throws -> Bounds3f {
                return try shape.worldBound(scene: scene)
        }

        func objectBound(scene: Scene) throws -> Bounds3f {
                return try shape.objectBound(scene: scene)
        }

        var shape: ShapeType
        var materialIndex: Int
        var mediumInterface: MediumInterface?
        var alpha: Real
        var idx: Int
}

typealias MaterialIndex = Int
let noMaterial = -1
