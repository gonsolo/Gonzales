struct GeometricPrimitive: Boundable, Intersectable, Material {

        func intersect(
                ray: Ray,
                tHit: inout FloatX,
                material: MaterialIndex,
                interaction: inout SurfaceInteraction
        ) throws {
                if alpha == 0 { return }
                // argument material is unused
                try shape.intersect(
                        ray: ray,
                        tHit: &tHit,
                        material: self.material,
                        interaction: &interaction)
                if interaction.valid {
                        interaction.mediumInterface = mediumInterface
                }
        }

        func worldBound() -> Bounds3f {
                return shape.worldBound()
        }

        func objectBound() -> Bounds3f {
                return shape.objectBound()
        }

        func getBSDF(interaction: Interaction) -> BSDF {
                return materials[material]!.getBSDF(interaction: interaction)
        }

        var shape: Shape
        var material: MaterialIndex
        var mediumInterface: MediumInterface?
        var alpha: FloatX
}

typealias MaterialIndex = Int
var materials: [Int: Material] = [:]
var materialCounter: MaterialIndex = 0
