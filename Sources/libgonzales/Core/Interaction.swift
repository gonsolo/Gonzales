protocol Interaction: Sendable {

        func spawnRay(target: Point) -> (ray: Ray, tHit: Real)
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
        let epsilon: Real = 0.0001
        return point + epsilon * direction
}

func spawnRay(from: Point, target: Point) -> (ray: Ray, tHit: Real) {
        let origin = offsetRayOrigin(point: from, direction: target - from)
        let direction: Vector = target - origin
        return (Ray(origin: origin, direction: direction), 1.0 - shadowEpsilon)
}

func offsetRayOrigin(point: Point, normal: Normal, direction: Vector) -> Point {
        let epsilon: Real = 0.0001
        let offset = epsilon * Vector(normal: normal)
        if dot(direction, normal) > 0 {
                return point + offset
        } else {
                return point + (-offset)
        }
}

func spawnRay(from: Point, normal: Normal, target: Point) -> (ray: Ray, tHit: Real) {
        let direction: Vector = target - from
        let origin = offsetRayOrigin(point: from, normal: normal, direction: direction)
        return (Ray(origin: origin, direction: direction), 1.0 - shadowEpsilon)
}

extension Interaction {

        func spawnRay(inDirection direction: Vector) -> Ray {
                let origin = offsetRayOrigin(point: position, normal: normal, direction: direction)
                return Ray(origin: origin, direction: direction)
        }

        func spawnRay(target: Point) -> (ray: Ray, tHit: Real) {
                let direction: Vector = target - position
                let origin = offsetRayOrigin(point: position, normal: normal, direction: direction)
                return (Ray(origin: origin, direction: direction), 1.0 - shadowEpsilon)
        }
}
