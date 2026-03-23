enum RgbSpectrumTexture: Sendable {

        // case checkerboard(Checkerboard)
        case constantTexture(ConstantTexture<RgbSpectrum>)
        case openImageIoTexture(OpenImageIOTexture)
        case ptex(Ptex)
        case scaledTexture(ScaledTextureRgb)
        // case rgbSpectrumMixTexture(RgbSpectrumMixTexture)
}
