struct Disk: Shape {

        init(objectToWorld: Transform, radius: Distance) {
                self.objectToWorld = objectToWorld
                self.radius = radius
        }

        func worldBound(scene: Scene) throws -> Bounds3f {
                return try objectToWorld * objectBound(scene: scene)
        }

        func objectBound(scene _: Scene) throws -> Bounds3f {
                throw RenderError.unimplemented(
                        function: #function, file: #filePath, line: #line, message: "")
        }

        func intersect(
                scene _: Scene,
                ray _: Ray,
                tHit _: inout Real
        ) throws -> Bool {
                throw RenderError.unimplemented(
                        function: #function, file: #filePath, line: #line, message: "")
        }

        func intersect(
                scene _: Scene,
                ray _: Ray,
                tHit _: inout Real
        ) throws -> SurfaceInteraction? {
                throw RenderError.unimplemented(
                        function: #function, file: #filePath, line: #line, message: "")
        }

        func area(scene _: Scene) throws -> Area {
                throw RenderError.unimplemented(
                        function: #function, file: #filePath, line: #line, message: "")
        }

        func sample<I: Interaction>(samples _: TwoRandomVariables, scene _: Scene) throws -> (
                interaction: I, pdf: Real
        ) {
                throw RenderError.unimplemented(
                        function: #function, file: #filePath, line: #line, message: "")
        }

        public var description: String {
                return "Disk"
        }

        func getObjectToWorld(scene _: Scene) -> Transform {
                return objectToWorld
        }

        let objectToWorld: Transform
        let radius: Distance
}

extension Disk {
        static func create(objectToWorld: Transform, parameters: ParameterDictionary) throws -> [ShapeType] {
                let radius = try parameters.findOneReal(called: "radius", else: 1.0)
                let shape = ShapeType.disk(Disk(objectToWorld: objectToWorld, radius: radius))
                return [shape]
        }
}
