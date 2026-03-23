enum FloatTexture: Sendable {
        case constantTexture(ConstantTexture<Real>)
        case openImageIoTexture(OpenImageIOTexture)
        case scaledTexture(ScaledTextureFloat)

        // case floatMixTexture(FloatMixTexture)
}
