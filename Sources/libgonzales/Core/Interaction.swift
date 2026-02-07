protocol Interaction: Sendable {

        func spawnRay(target: Point) -> (ray: Ray, tHit: FloatX)
        func spawnRay(inDirection direction: Vector) -> Ray

        var dpdu: Vector { get }
        var faceIndex: Int { get }
        var normal: Normal { get }
        var position: Point { get }
        var shadingNormal: Normal { get }
        var uvCoordinates: Point2f { get }
        var outgoing: Vector { get }
}

func offsetRayOrigin(point: Point, direction: Vector) -> Point {
        let epsilon: FloatX = 0.0001
        return Point(point + epsilon * direction)
}

func spawnRay(from: Point, target: Point) -> (ray: Ray, tHit: FloatX) {
        let origin = offsetRayOrigin(point: from, direction: target - from)
        let direction: Vector = target - origin
        return (Ray(origin: origin, direction: direction), FloatX(1.0) - shadowEpsilon)
}

extension Interaction {

        func spawnRay(inDirection direction: Vector) -> Ray {
                let origin = offsetRayOrigin(point: position, direction: direction)
                return Ray(origin: origin, direction: direction)
        }

        func spawnRay(target: Point) -> (ray: Ray, tHit: FloatX) {
                let origin = offsetRayOrigin(point: position, direction: target - position)
                let direction: Vector = target - origin
                return (Ray(origin: origin, direction: direction), FloatX(1.0) - shadowEpsilon)
        }

        func offsetRayOrigin(point: Point, direction: Vector) -> Point {
                let epsilon: FloatX = 0.0001
                return Point(point + epsilon * direction)
        }
}
