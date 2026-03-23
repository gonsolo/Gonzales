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
		switch self {
		case .geometricPrimitive(let value): return try value.getIntersectionData(scene: scene, ray: worldRay, tHit: &tHit, data: &data)
		case .triangle(let value): return try value.getIntersectionData(scene: scene, ray: worldRay, tHit: &tHit, data: &data)
		case .transformedPrimitive(let value): return try value.getIntersectionData(scene: scene, ray: worldRay, tHit: &tHit, data: &data)
		case .areaLight(let value): return try value.getIntersectionData(scene: scene, ray: worldRay, tHit: &tHit, data: &data)
		}
	}

	func computeSurfaceInteraction(
		scene: Scene,
		data: TriangleIntersection,
		worldRay: Ray
	) throws -> SurfaceInteraction? {
		switch self {
		case .geometricPrimitive(let value): return try value.computeSurfaceInteraction(scene: scene, data: data, worldRay: worldRay)
		case .triangle(let value): return value.computeSurfaceInteraction(scene: scene, data: data, worldRay: worldRay)
		case .transformedPrimitive(let value): return try value.computeSurfaceInteraction(scene: scene, data: data, worldRay: worldRay)
		case .areaLight(let value): return try value.computeSurfaceInteraction(scene: scene, data: data, worldRay: worldRay)
		}
	}

	func intersect(
		scene: Scene,
		ray: Ray,
		tHit: inout Real
	) throws -> SurfaceInteraction? {
		switch self {
		case .geometricPrimitive(let value): return try value.intersect(scene: scene, ray: ray, tHit: &tHit)
		case .triangle(let value): return try value.intersect(scene: scene, ray: ray, tHit: &tHit)
		case .transformedPrimitive(let value): return try value.intersect(scene: scene, ray: ray, tHit: &tHit)
		case .areaLight(let value): return try value.intersect(scene: scene, ray: ray, tHit: &tHit)
		}
	}

	func worldBound(scene: Scene) throws -> Bounds3f {
		switch self {
		case .geometricPrimitive(let value): return try value.worldBound(scene: scene)
		case .triangle(let value): return value.worldBound(scene: scene)
		case .transformedPrimitive(let value): return value.worldBound(scene: scene)
		case .areaLight(let value): return try value.worldBound(scene: scene)
		}
	}

	func objectBound(scene: Scene) throws -> Bounds3f {
		switch self {
		case .geometricPrimitive(let value): return try value.objectBound(scene: scene)
		case .triangle(let value): return value.objectBound(scene: scene)
		case .transformedPrimitive(let value): return try value.objectBound(scene: scene)
		case .areaLight(let value): return try value.objectBound(scene: scene)
		}
	}
}
