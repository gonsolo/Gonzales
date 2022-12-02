struct GeometricPrimitive: Boundable, Intersectable, Material {

        init(shape: Shape, material: MaterialIndex) {
                self.shape = shape
                self.material = material
        }

        func intersect(
                ray: Ray,
                tHit: inout FloatX,
                material: MaterialIndex,
                interaction: inout SurfaceInteraction
        ) throws {
                // argument material is unused
                try shape.intersect(
                        ray: ray,
                        tHit: &tHit,
                        material: self.material,
                        interaction: &interaction)
        }

        func worldBound() -> Bounds3f {
                return shape.worldBound()
        }

        func objectBound() -> Bounds3f {
                return shape.objectBound()
        }

        func computeScatteringFunctions(interaction: Interaction) -> (BSDF, BSSRDF?) {
                return materials[material]!.computeScatteringFunctions(interaction: interaction)
        }

        var shape: Shape
        var material: MaterialIndex
}

typealias MaterialIndex = Int
var materials: [Int: Material] = [:]
var materialCounter: MaterialIndex = 0
