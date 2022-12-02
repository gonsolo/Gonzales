/// A type that can calculate the visibility from a point to another.

struct Visibility {

        func unoccluded() throws -> Bool {
                var (ray, tHit) = from.spawnRay(to: to.position)
                var interaction = SurfaceInteraction()
                try scene.intersect(ray: ray, tHit: &tHit, interaction: &interaction)
                if interaction.valid {
                        return false
                } else {
                        return true
                }
        }

        let from: Interaction
        let to: Interaction
}
