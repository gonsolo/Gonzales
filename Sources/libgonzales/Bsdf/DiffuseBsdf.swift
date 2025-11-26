public struct DiffuseBsdf: GlobalBsdf {

        var reflectance: RgbSpectrum = white
        public let bsdfFrame: BsdfFrame

        public init(reflectance: RgbSpectrum, bsdfFrame: BsdfFrame) {
                self.reflectance = reflectance
                self.bsdfFrame = bsdfFrame
        }
}

extension DiffuseBsdf {

        public func evaluateLocal(wo _: Vector, wi _: Vector) -> RgbSpectrum {
                return reflectance / FloatX.pi
        }

        public func albedo() -> RgbSpectrum { return reflectance }

}
