/// A type that can calculate the visibility from a point to another.

struct Visibility {

        func unoccluded() throws -> Bool {
                //var (ray, tHit) = from.spawnRay(to: to.position)
                //return try scene.intersect(ray: ray, tHit: &tHit) == nil
                let interaction = try getOccluder()
                if interaction.valid {
                        return false
                } else {
                        return true
                }
        }

        func getOccluder() throws -> SurfaceInteraction {
                var (ray, tHit) = from.spawnRay(to: to.position)
                return try scene.intersect(ray: ray, tHit: &tHit)
        }

        var from: Interaction
        var to: Interaction
}
