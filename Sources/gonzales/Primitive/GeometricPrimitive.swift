struct GeometricPrimitive: Boundable, Intersectable {

        func getIntersectionData(
                ray worldRay: Ray,
                tHit: inout FloatX
        ) throws -> IntersectablePrimitiveIntersection {
                if alpha == 0 { return .triangle(nil) }
                return try shape.getIntersectionData(
                        ray: worldRay,
                        tHit: &tHit)
        }

        func computeSurfaceInteraction(
                data: IntersectablePrimitiveIntersection,
                worldRay: Ray,
                interaction: inout SurfaceInteraction
        ) {
                if alpha == 0 { return }
                shape.computeSurfaceInteraction(
                        data: data,
                        worldRay: worldRay,
                        interaction: &interaction)
                if interaction.valid {
                        interaction.materialIndex = materialIndex
                        //interaction.mediumInterface = mediumInterface
                }
        }

        func intersect(
                ray: Ray,
                tHit: inout FloatX
        ) throws -> Bool {
                if alpha == 0 { return false }
                return try shape.intersect(
                        ray: ray,
                        tHit: &tHit)
        }

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
                        //interaction.material = material
                        interaction.materialIndex = materialIndex
                        //interaction.mediumInterface = mediumInterface
                }
        }

        func worldBound() async -> Bounds3f {
                return await shape.worldBound()
        }

        func objectBound() async -> Bounds3f {
                return await shape.objectBound()
        }

        //func getBsdf(interaction: SurfaceInteraction) -> GlobalBsdfType {
        //        return materials[materialIndex].getBsdf(interaction: interaction)
        //}

        //var shape: any Shape
        var shape: ShapeType
        //var material: Material
        var materialIndex: Int
        var mediumInterface: MediumInterface?
        var alpha: FloatX
        var idx: Int
}

typealias MaterialIndex = Int
let noMaterial = -1

@MainActor
var materials: [Material] = []
