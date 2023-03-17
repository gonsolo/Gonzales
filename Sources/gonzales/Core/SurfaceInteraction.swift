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

struct SurfaceInteraction: Interaction {

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
                material: MaterialIndex = -1,
                mediumInterface: MediumInterface? = nil
        ) {
                self.valid = valid
                self.position = position
                self.normal = normal
                self.shadingNormal = shadingNormal
                self.wo = wo
                self.dpdu = dpdu
                self.uv = uv
                self.faceIndex = faceIndex
                self.material = material
                self.mediumInterface = mediumInterface
        }

        init(_ other: SurfaceInteraction) {
                self.valid = other.valid
                self.position = other.position
                self.normal = other.normal
                self.shadingNormal = other.shadingNormal
                self.wo = other.wo
                self.dpdu = other.dpdu
                self.uv = other.uv
                self.faceIndex = other.faceIndex
                self.material = other.material
                self.mediumInterface = other.mediumInterface
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

        var valid: Bool
        var position: Point
        var normal: Normal
        var shadingNormal: Normal
        var wo: Vector
        var dpdu: Vector
        var uv: Point2F
        var faceIndex: Int
        var areaLight: AreaLight?
        var material: MaterialIndex
        var mediumInterface: MediumInterface?
}

extension SurfaceInteraction: CustomStringConvertible {
        var description: String {
                return "[pos: \(position) n: \(normal) wo: \(wo) ]"
        }
}
