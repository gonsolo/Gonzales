struct Disk: Shape {

        init(objectToWorld: Transform, radius: FloatX) {
                self.objectToWorld = objectToWorld
                self.radius = radius
        }

        func worldBound(scene: Scene) -> Bounds3f {
                return objectToWorld * objectBound(scene: scene)
        }

        func objectBound(scene _: Scene) -> Bounds3f {
                // TODO
                return Bounds3f()
        }

        func intersect(
                scene _: Scene,
                ray _: Ray,
                tHit _: inout FloatX
        ) throws -> Bool {
                unimplemented()
        }

        func intersect(
                scene _: Scene,
                ray _: Ray,
                tHit _: inout FloatX,
                interaction _: inout SurfaceInteraction
        ) throws {
                unimplemented()
        }

        func area(scene _: Scene) -> FloatX {
                unimplemented()
        }

        func sample<I: Interaction>(samples _: TwoRandomVariables, scene _: Scene) -> (
                interaction: I, pdf: FloatX
        ) {
                unimplemented()
        }

        public var description: String {
                return "Disk"
        }

        static func statistics() {
                print("TODO")
        }

        func getObjectToWorld(scene _: Scene) -> Transform {
                return objectToWorld
        }

        let objectToWorld: Transform
        let radius: FloatX
}

func createDiskShape(objectToWorld: Transform, parameters: ParameterDictionary) throws -> [ShapeType] {
        let radius = try parameters.findOneFloatX(called: "radius", else: 1.0)
        let shape = ShapeType.disk(Disk(objectToWorld: objectToWorld, radius: radius))
        return [shape]
}
