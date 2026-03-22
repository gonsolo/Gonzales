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
                let interaction = try shape.computeSurfaceInteraction(
                        scene: scene,
                        data: data,
                        worldRay: worldRay)
                if var unwrapped = interaction {
                        unwrapped.materialIndex = materialIndex
                        if reverseOrientation {
                                unwrapped.normal = -unwrapped.normal
                                unwrapped.shadingNormal = -unwrapped.shadingNormal
                        }
                        return unwrapped
                }
                return nil
        }

        func intersect(
                scene: Scene,
                ray: Ray,
                tHit: inout Real
        ) throws -> SurfaceInteraction? {
                if alpha == 0 { return nil }
                let interaction = try shape.intersect(
                        scene: scene,
                        ray: ray,
                        tHit: &tHit)
                if var unwrapped = interaction {
                        unwrapped.materialIndex = materialIndex
                        if reverseOrientation {
                                unwrapped.normal = -unwrapped.normal
                                unwrapped.shadingNormal = -unwrapped.shadingNormal
                        }
                        return unwrapped
                }
                return nil
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
        var reverseOrientation: Bool
        var idx: Int
}

typealias MaterialIndex = Int
let noMaterial = -1
