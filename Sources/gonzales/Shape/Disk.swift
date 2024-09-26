final class Disk: Shape {

        init(objectToWorld: Transform, radius: FloatX) {
                self.objectToWorld = objectToWorld
                self.radius = radius
        }

        @MainActor
        func worldBound() -> Bounds3f {
                return objectToWorld * objectBound()
        }

        @MainActor
        func objectBound() -> Bounds3f {
                // TODO
                return Bounds3f()
        }

        @MainActor
        func intersect(
                ray worldRay: Ray,
                tHit: inout FloatX,
                interaction: inout SurfaceInteraction
        ) async throws {
                // TODO
        }

        func area() -> FloatX {
                fatalError("Not implemented")
        }

        func sample(u: TwoRandomVariables) -> (interaction: Interaction, pdf: FloatX) {
                // TODO
                return (SurfaceInteraction(), 0)
        }

        public var description: String {
                return "Disk"
        }

        static func statistics() {
                print("TODO")
        }

        var objectToWorld: Transform
        var radius: FloatX
}

func createDiskShape(objectToWorld: Transform, parameters: ParameterDictionary) throws -> [Shape] {
        let radius = try parameters.findOneFloatX(called: "radius", else: 1.0)
        let disk = Disk(objectToWorld: objectToWorld, radius: radius)
        return [disk]
}
