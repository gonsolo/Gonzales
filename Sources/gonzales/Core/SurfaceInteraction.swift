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
                primitive: (Boundable & Intersectable)? = nil,
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
                self.primitive = primitive
                self.material = material
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
                self.primitive = other.primitive
                self.material = other.material
        }

        let valid: Bool
        let position: Point
        let normal: Normal
        let shadingNormal: Normal
        let wo: Vector
        let dpdu: Vector
        let uv: Point2F
        let faceIndex: Int

        let primitive: (Boundable & Intersectable)?
        let material: MaterialIndex
}

extension SurfaceInteraction: CustomStringConvertible {
        var description: String {
                return "[pos: \(position) n: \(normal) wo: \(wo) ]"
        }
}
