import DevirtualizeMacro

/// A type that can be intersected by a ray.

protocol Intersectable {

        func intersect(
                scene: Scene,
                ray: Ray,
                tHit: inout Real
        ) throws -> SurfaceInteraction?
}

enum IntersectablePrimitive: Intersectable, Sendable, Boundable {
        case geometricPrimitive(GeometricPrimitive)
        case triangle(Triangle)
        case transformedPrimitive(TransformedPrimitive)
        case areaLight(AreaLight)

        func getIntersectionData(
                scene: Scene,
                ray worldRay: Ray,
                tHit: inout Real,
                data: inout TriangleIntersection
        ) throws -> Bool {
                return try #dispatchIntersectable(primitive: self) { (p: Triangle) in
                        try p.getIntersectionData(scene: scene, ray: worldRay, tHit: &tHit, data: &data)
                }
        }

        func computeSurfaceInteraction(
                scene: Scene,
                data: TriangleIntersection,
                worldRay: Ray
        ) throws -> SurfaceInteraction? {
                return try #dispatchIntersectable(primitive: self) { (p: Triangle) in
                        try p.computeSurfaceInteraction(scene: scene, data: data, worldRay: worldRay)
                }
        }

        func intersect(
                scene: Scene,
                ray: Ray,
                tHit: inout Real
        ) throws -> SurfaceInteraction? {
                return try #dispatchIntersectable(primitive: self) { (p: Triangle) in
                        try p.intersect(scene: scene, ray: ray, tHit: &tHit)
                }
        }

        func worldBound(scene: Scene) throws -> Bounds3f {
                return try #dispatchIntersectable(primitive: self) { (p: Triangle) in
                        try p.worldBound(scene: scene)
                }
        }

        func objectBound(scene: Scene) throws -> Bounds3f {
                return try #dispatchIntersectable(primitive: self) { (p: Triangle) in
                        try p.objectBound(scene: scene)
                }
        }
}
