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
                mediumInterface: MediumInterface? = nil,
                bsdf: GlobalBsdf = DummyBsdf()
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
                self.bsdf = bsdf
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
                self.bsdf = other.bsdf
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

        var bsdf: GlobalBsdf
}

extension SurfaceInteraction: CustomStringConvertible {
        var description: String {
                return "[pos: \(position) n: \(normal) wo: \(wo) ]"
        }
}
