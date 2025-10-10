final class Disk: Shape {

        init(objectToWorld: Transform, radius: FloatX) {
                self.objectToWorld = objectToWorld
                self.radius = radius
        }

        func worldBound() -> Bounds3f {
                return objectToWorld * objectBound()
        }

        func objectBound() -> Bounds3f {
                // TODO
                return Bounds3f()
        }

        func intersect(
                ray worldRay: Ray,
                tHit: inout FloatX
        ) throws -> Bool {
                unimplemented()
        }

        func intersect(
                ray worldRay: Ray,
                tHit: inout FloatX,
                interaction: inout SurfaceInteraction
        ) throws {
                unimplemented()
        }

        func area() -> FloatX {
                unimplemented()
        }

        func sample(u: TwoRandomVariables) -> (interaction: any Interaction, pdf: FloatX) {
                // TODO
                return (SurfaceInteraction(), 0)
        }

        public var description: String {
                return "Disk"
        }

        static func statistics() {
                print("TODO")
        }

        let objectToWorld: Transform
        let radius: FloatX
}

func createDiskShape(objectToWorld: Transform, parameters: ParameterDictionary) throws -> [ShapeType] {
        let radius = try parameters.findOneFloatX(called: "radius", else: 1.0)
        let shape = ShapeType.disk(Disk(objectToWorld: objectToWorld, radius: radius))
        return [shape]
}
