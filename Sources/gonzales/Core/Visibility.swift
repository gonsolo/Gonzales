/// A type that can calculate the visibility from a point to another.

func offsetRayOrigin(point: Point, direction: Vector) -> Point {
    let epsilon: FloatX = 0.0001
    return Point(point + epsilon * direction)
}

func spawnRay(from: Point, to: Point) -> (ray: Ray, tHit: FloatX) {
    let origin = offsetRayOrigin(point: from, direction: to - from)
    let direction: Vector = to - origin
    return (Ray(origin: origin, direction: direction), FloatX(1.0) - shadowEpsilon)
}

struct Visibility {

        func unoccluded(scene: Scene) throws -> Bool {
                var (ray, tHit) = spawnRay(from: from, to: to)
                var interaction = SurfaceInteraction()
                try scene.intersect(
                        ray: ray,
                        tHit: &tHit,
                        interaction: &interaction)
                if interaction.valid {
                        return false
                } else {
                        return true
                }
        }

        let from: Point
        let to: Point
}
