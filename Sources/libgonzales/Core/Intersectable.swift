/// A type that can be intersected by a ray.

protocol Intersectable {

	func intersect(
		scene: Scene,
		ray: Ray,
		tHit: inout FloatX
	) throws -> SurfaceInteraction?
}

// enum IntersectablePrimitiveIntersection {
//        case triangle(TriangleIntersection?)
// }

enum IntersectablePrimitive: Intersectable, Sendable {
	case geometricPrimitive(GeometricPrimitive)
	case triangle(Triangle)
	case transformedPrimitive(TransformedPrimitive)
	case areaLight(AreaLight)

	func getIntersectionData(
		scene: Scene,
		ray worldRay: Ray,
		tHit: inout FloatX,
		data: inout TriangleIntersection
	) throws -> Bool {
		switch self {
		case .areaLight(let areaLight):
			return try areaLight.getIntersectionData(
				scene: scene,
				ray: worldRay,
				tHit: &tHit,
				data: &data)
		case .geometricPrimitive(let geometricPrimitive):
			return try geometricPrimitive.getIntersectionData(
				scene: scene,
				ray: worldRay,
				tHit: &tHit,
				data: &data)
		case .triangle(let triangle):
			return try triangle.getIntersectionData(
				scene: scene,
				ray: worldRay,
				tHit: &tHit,
				data: &data)
		case .transformedPrimitive:
			unimplemented()
		}

	}

	func computeSurfaceInteraction(
		scene: Scene,
		data: TriangleIntersection?,
		worldRay: Ray
	) -> SurfaceInteraction? {
		switch self {
		case .geometricPrimitive(let geometricPrimitive):
			return geometricPrimitive.computeSurfaceInteraction(
				scene: scene,
				data: data,
				worldRay: worldRay)
		case .triangle(let triangle):
			return triangle.computeSurfaceInteraction(
				scene: scene,
				data: data,
				worldRay: worldRay)
		case .transformedPrimitive:
			unimplemented()
		case .areaLight(let areaLight):
			return areaLight.computeSurfaceInteraction(
				scene: scene,
				data: data,
				worldRay: worldRay)
		}
	}

	func intersect(
		scene: Scene,
		ray: Ray,
		tHit: inout FloatX
	) throws -> SurfaceInteraction? {
		switch self {
		case .areaLight(let areaLight):
			return try areaLight.intersect(
				scene: scene,
				ray: ray,
				tHit: &tHit)
		case .geometricPrimitive(let geometricPrimitive):
			return try geometricPrimitive.intersect(
				scene: scene,
				ray: ray,
				tHit: &tHit)
		case .triangle(let triangle):
			return try triangle.intersect(
				scene: scene,
				ray: ray,
				tHit: &tHit)
		case .transformedPrimitive(let transformedPrimitive):
			return try transformedPrimitive.intersect(
				scene: scene,
				ray: ray,
				tHit: &tHit)
		}
	}
}
