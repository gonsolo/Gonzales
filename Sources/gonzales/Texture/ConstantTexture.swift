final class ConstantTexture<T: TextureEvaluation>: Texture {

        init(value: T) {
                self.value = value
        }

        func evaluate(at: Interaction) -> TextureEvaluation {
                return value
        }

        var value: T
}

extension ConstantTexture: SpectrumTexture where T == Spectrum {
        func evaluateSpectrum(at interaction: Interaction) -> Spectrum {
                return evaluate(at: interaction) as! Spectrum
        }
}

extension ConstantTexture: FloatTexture where T == FloatX {
        func evaluateFloat(at interaction: Interaction) -> FloatX {
                return evaluate(at: interaction) as! FloatX
        }
}
