/// A type that can calculate the visibility from a point to another.

struct Visibility {

        func occluded(scene: Scene) throws -> Bool {
                var (ray, tHit) = spawnRay(from: from, to: to)
                return try scene.intersect(
                        ray: ray,
                        tHit: &tHit)
        }

        let from: Point
        let to: Point
}
