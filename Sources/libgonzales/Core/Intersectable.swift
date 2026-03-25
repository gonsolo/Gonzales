import DevirtualizeMacro

/// A type that can be intersected by a ray.

protocol Intersectable {

        func intersect(
                scene: Scene,
                ray: Ray,
                tHit: inout Real
        ) -> SurfaceInteraction?
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
        ) -> Bool {
                return #dispatchIntersectableNoThrow(primitive: self) { (typedPrimitive: Triangle) in
                        typedPrimitive.getIntersectionData(scene: scene, ray: worldRay, tHit: &tHit, data: &data)
                }
        }

        func computeSurfaceInteraction(
                scene: Scene,
                data: TriangleIntersection,
                worldRay: Ray
        ) -> SurfaceInteraction? {
                return #dispatchIntersectableNoThrow(primitive: self) { (typedPrimitive: Triangle) in
                        typedPrimitive.computeSurfaceInteraction(scene: scene, data: data, worldRay: worldRay)
                }
        }

        func intersect(
                scene: Scene,
                ray: Ray,
                tHit: inout Real
        ) -> SurfaceInteraction? {
                return #dispatchIntersectableNoThrow(primitive: self) { (typedPrimitive: Triangle) in
                        typedPrimitive.intersect(scene: scene, ray: ray, tHit: &tHit)
                }
        }

        func worldBound(scene: Scene) -> Bounds3f {
                return #dispatchIntersectableNoThrow(primitive: self) { (typedPrimitive: Triangle) in
                        typedPrimitive.worldBound(scene: scene)
                }
        }

        func objectBound(scene: Scene) -> Bounds3f {
                return #dispatchIntersectableNoThrow(primitive: self) { (typedPrimitive: Triangle) in
                        typedPrimitive.objectBound(scene: scene)
                }
        }
}
