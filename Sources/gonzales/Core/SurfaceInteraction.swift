typealias SurfaceInteraction = Interaction

struct Interaction {

        init(
                valid: Bool = false,
                position: Point = Point(),
                normal: Normal = Normal(),
                shadingNormal: Normal = Normal(),
                wo: Vector = Vector(),
                dpdu: Vector = Vector(),
                uv: Point2F = Point2F(),
                faceIndex: Int = 0,
                areaLight: AreaLight? = nil,
                material: MaterialIndex = -1
        ) {
                self.valid = valid
                self.position = position
                self.normal = normal
                self.shadingNormal = shadingNormal
                self.wo = wo
                self.dpdu = dpdu
                self.uv = uv
                self.faceIndex = faceIndex
                self.areaLight = areaLight
                self.material = material
        }

        init(_ other: Interaction) {
                self.valid = other.valid
                self.position = other.position
                self.normal = other.normal
                self.shadingNormal = other.shadingNormal
                self.wo = other.wo
                self.dpdu = other.dpdu
                self.uv = other.uv
                self.faceIndex = other.faceIndex
                self.areaLight = other.areaLight
                self.material = other.material
        }

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

        let valid: Bool
        let position: Point
        let normal: Normal
        let shadingNormal: Normal
        let wo: Vector
        let dpdu: Vector
        let uv: Point2F
        let faceIndex: Int
        let areaLight: AreaLight?
        let material: MaterialIndex
}

extension Interaction: CustomStringConvertible {
        var description: String {
                return "[pos: \(position) n: \(normal) wo: \(wo) ]"
        }
}
