/// A type that can be intersected by a ray.

protocol Intersectable {

	func intersect(
		scene: Scene,
		ray: Ray,
		tHit: inout Real
	) throws -> SurfaceInteraction?
}

// enum IntersectablePrimitiveIntersection {
//        case triangle(TriangleIntersection?)
// }

enum IntersectablePrimitive: Intersectable, Sendable, Boundable {
	case geometricPrimitive(GeometricPrimitive)
	case triangle(Triangle)
	case transformedPrimitive(TransformedPrimitive)
	case areaLight(AreaLight)
}
