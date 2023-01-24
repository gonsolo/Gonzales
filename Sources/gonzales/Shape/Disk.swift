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
                tHit: inout FloatX,
                material: MaterialIndex,
                interaction: inout SurfaceInteraction
        ) throws {
                //let ray = worldToObject * worldRay
                // TODO
        }

        func area() -> FloatX {
                fatalError("Not implemented")
        }

        func sample(u: Point2F) -> (interaction: Interaction, pdf: FloatX) {
                // TODO
                return (Interaction(), 0)
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
