public protocol DistributionModel {
        func evaluateDistributionFunction(outgoing: Vector, incident: Vector, normal: Normal) -> RgbSpectrum
        func sampleDistributionFunction(outgoing: Vector, normal: Normal, sampler: inout Sampler)
                -> BsdfSample
        func evaluateProbabilityDensity(outgoing: Vector, incident: Vector) -> FloatX
}

struct SurfaceInteraction: Interaction, Sendable {

        var valid = false
        var position = Point()
        var normal = Normal()
        var shadingNormal = Normal()
        var outgoing = Vector()
        var dpdu = Vector()
        var uv = Point2f()
        var faceIndex = 0
        var areaLight: AreaLight?
        var materialIndex: MaterialIndex = 0
        // var mediumInterface: MediumInterface? = nil
}

extension SurfaceInteraction: CustomStringConvertible {
        var description: String {
                return
                        "[" + "valid: \(valid) " + "pos: \(position) " + "n: \(normal) "
                        + "shadingNormal: \(shadingNormal) " + "outgoing: \(outgoing) " + "dpdu: \(dpdu) " + "uv: \(uv) "
                        + "faceIndex: \(faceIndex) " + "areaLight: \(areaLight as Optional) "
                        + "materialIndex: \(materialIndex) "  // + "mediumInterface: \(mediumInterface as Optional) "
                        + "]"
        }
}

extension SurfaceInteraction: Equatable {
        static func == (lhs: SurfaceInteraction, rhs: SurfaceInteraction) -> Bool {
                return
                        lhs.valid == rhs.valid && lhs.position == rhs.position && lhs.normal == rhs.normal
                        && lhs.shadingNormal == rhs.shadingNormal && lhs.outgoing == rhs.outgoing && lhs.dpdu == rhs.dpdu
                        && lhs.uv == rhs.uv && lhs.faceIndex == rhs.faceIndex
                        && lhs.areaLight == rhs.areaLight  // &&
                // lhs.material == rhs.material &&
                // lhs.mediumInterface == rhs.mediumInterface &&
                // lhs.bsdf == rhs.bsdf
        }
}
