/// A type that can calculate the visibility from a point to another.

struct Visibility {

        func occluded(scene: Scene, accelerator: Accelerator) throws -> Bool {
                var (ray, tHit) = spawnRay(from: from, target: target)
                while true {
                        guard let interaction = try accelerator.intersect(
                                scene: scene,
                                ray: ray,
                                tHit: &tHit) else {
                                return false
                        }
                        if interaction.materialIndex < 0 {
                                let next = interaction.spawnRay(target: target)
                                ray = next.ray
                                tHit = next.tHit
                                continue
                        }
                        return true
                }
        }

        let from: Point
        let target: Point
}
