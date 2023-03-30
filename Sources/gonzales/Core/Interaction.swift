protocol Interaction {

        func spawnRay(to: Point) -> (ray: Ray, tHit: FloatX)
        func spawnRay(inDirection direction: Vector) -> Ray

        var dpdu: Vector { get }
        var faceIndex: Int { get }
        var normal: Normal { get }
        var position: Point { get }
        var shadingNormal: Normal { get }
        var uv: Point2F { get }
        var wo: Vector { get }
}

extension Interaction {

        func spawnRay(inDirection direction: Vector) -> Ray {
                let origin = offsetRayOrigin(point: position, direction: direction)
                return Ray(origin: origin, direction: direction)
        }

        func spawnRay(to: Point) -> (ray: Ray, tHit: FloatX) {
                let origin = offsetRayOrigin(point: position, direction: to - position)
                let direction: Vector = to - origin
                return (Ray(origin: origin, direction: direction), FloatX(1.0) - shadowEpsilon)
        }

        func offsetRayOrigin(point: Point, direction: Vector) -> Point {
                let epsilon: FloatX = 0.0001
                return Point(point + epsilon * direction)
        }
}
