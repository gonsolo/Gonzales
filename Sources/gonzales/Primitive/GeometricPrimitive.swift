struct GeometricPrimitive: Boundable, Intersectable {

        @MainActor
        func intersect(
                ray: Ray,
                tHit: inout FloatX,
                interaction: inout SurfaceInteraction
        ) async throws {
                if alpha == 0 { return }
                try await shape.intersect(
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

        @MainActor
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

@MainActor
var materials: [Material] = []
