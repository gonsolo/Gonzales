struct GeometricPrimitive: Boundable, Intersectable {

        func getIntersectionData(
                scene: Scene,
                ray worldRay: Ray,
                tHit: inout FloatX,
                data: inout TriangleIntersection
        ) throws -> Bool {
                if alpha == 0 { return false }
                return try shape.getIntersectionData(
                        scene: scene,
                        ray: worldRay,
                        tHit: &tHit,
                        data: &data)
        }

        func computeSurfaceInteraction(
                scene: Scene,
                data: TriangleIntersection?,
                worldRay: Ray,
                interaction: inout SurfaceInteraction
        ) {
                if alpha == 0 { return }
                shape.computeSurfaceInteraction(
                        scene: scene,
                        data: data,
                        worldRay: worldRay,
                        interaction: &interaction)
                if interaction.valid {
                        interaction.materialIndex = materialIndex
                        //interaction.mediumInterface = mediumInterface
                }
        }

        func intersect(
                scene: Scene,
                ray: Ray,
                tHit: inout FloatX
        ) throws -> Bool {
                if alpha == 0 { return false }
                return try shape.intersect(
                        scene: scene,
                        ray: ray,
                        tHit: &tHit)
        }

        func intersect(
                scene: Scene,
                ray: Ray,
                tHit: inout FloatX,
                interaction: inout SurfaceInteraction
        ) throws {
                if alpha == 0 { return }
                try shape.intersect(
                        scene: scene,
                        ray: ray,
                        tHit: &tHit,
                        interaction: &interaction)
                if interaction.valid {
                        //interaction.material = material
                        interaction.materialIndex = materialIndex
                        //interaction.mediumInterface = mediumInterface
                }
        }

        func worldBound(scene: Scene) async -> Bounds3f {
                return shape.worldBound(scene: scene)
        }

        func objectBound(scene: Scene) async -> Bounds3f {
                return shape.objectBound(scene: scene)
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
