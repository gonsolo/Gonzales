struct Ray: Sendable {

        init() {
                self.init(
                        origin: Point3(x: 0, y: 0, z: 0),
                        direction: Vector3(x: 0, y: 0, z: 1),
                        cameraSample: CameraSample()
                )
        }

        init(origin: Point, direction: Vector, cameraSample: CameraSample = CameraSample()) {
                self.origin = origin
                self.direction = direction
                self.inverseDirection = Vector(v: 1) / direction
                self.medium = nil
                self.cameraSample = cameraSample
        }

        func getPointFor(parameter t: FloatX) -> Point {
                return origin + t * direction
        }

        let origin: Point
        let direction: Vector
        let inverseDirection: Vector
        var medium: Medium?
        let cameraSample: CameraSample
}

extension Ray: CustomStringConvertible {
        var description: String {
                return "Ray [ o: \(origin) d: \(direction)]"
        }
}
