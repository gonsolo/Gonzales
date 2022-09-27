struct Ray {

        init() {
                self.init(origin: Point3(x: 0, y: 0, z: 0), direction: Vector3(x: 0, y: 0, z: 1))
        }

        init(origin: Point, direction: Vector) {
                self.origin = origin
                self.direction = direction
        }

        func getPointFor(parameter t: FloatX) -> Point {
                return origin + t * direction
        }

        var origin: Point
        var direction: Vector
}

extension Ray: CustomStringConvertible {
        var description: String {
                return "Ray [ o: \(origin) d: \(direction)]"
        }
}
