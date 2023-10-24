struct GeometricPrimitive: Boundable, Intersectable {

        func intersect(
                ray: Ray,
                tHit: inout FloatX,
                interaction: inout SurfaceInteraction
        ) throws {
                if alpha == 0 { return }
                try shape.intersect(
                        ray: ray,
                        tHit: &tHit,
                        interaction: &interaction)
                if interaction.valid {
                        interaction.material = material
                        interaction.mediumInterface = mediumInterface
                }
        }

        func worldBound() -> Bounds3f {
                return shape.worldBound()
        }

        func objectBound() -> Bounds3f {
                return shape.objectBound()
        }

        func getBsdf(interaction: Interaction) -> GlobalBsdf {
                return materials[material].getBsdf(interaction: interaction)
        }

        var shape: Shape
        var material: MaterialIndex
        var mediumInterface: MediumInterface?
        var alpha: FloatX
}

typealias MaterialIndex = Int
let noMaterial = -1
var materials: [Material] = []
