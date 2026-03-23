struct ConstantTexture<T: TextureEvaluation> {

        func evaluate(at _: any Interaction, arena _: TextureArena) -> any TextureEvaluation {
                return value
        }

        var value: T
}

extension ConstantTexture where T == RgbSpectrum {

        func evaluateRgbSpectrum(at _: any Interaction, arena _: TextureArena) -> RgbSpectrum {
                return value
        }
}

extension ConstantTexture where T == Real {
        func evaluateFloat(at _: any Interaction, arena _: TextureArena) -> Real {
                return value
        }
}

extension ConstantTexture: CustomStringConvertible {
        var description: String {
                return "\(value)"
        }
}
