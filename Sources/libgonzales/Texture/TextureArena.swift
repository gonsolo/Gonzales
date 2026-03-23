struct TextureArena: Sendable {
        var floatTextures: [FloatTexture] = []
        var rgbTextures: [RgbSpectrumTexture] = []

        mutating func appendFloat(_ texture: FloatTexture) -> Int {
                floatTextures.append(texture)
                return floatTextures.count - 1
        }

        mutating func appendRgb(_ texture: RgbSpectrumTexture) -> Int {
                rgbTextures.append(texture)
                return rgbTextures.count - 1
        }
}
