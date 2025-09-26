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

        func worldBound() async -> Bounds3f {
                return await shape.worldBound()
        }

        func objectBound() async -> Bounds3f {
                return await shape.objectBound()
        }

        func getBsdf(interaction: any Interaction) -> GlobalBsdfType {
                return material.getBsdf(interaction: interaction)
        }

        //var shape: any Shape
        var shape: ShapeType
        var material: Material
        var mediumInterface: MediumInterface?
        var alpha: FloatX
}

typealias MaterialIndex = Int
let noMaterial = -1

@MainActor
var materials: [Material] = []
