enum FloatTexture: Sendable {
        case constantTexture(ConstantTexture<Real>)
        case openImageIoTexture(OpenImageIOTexture)
        case scaledTexture(ScaledTextureFloat)

        // case floatMixTexture(FloatMixTexture)

        func evaluateFloat(at interaction: any Interaction, arena: TextureArena) -> Real {
                switch self {
                case .constantTexture(let value): return value.evaluateFloat(at: interaction, arena: arena)
                case .openImageIoTexture(let value): return value.evaluateFloat(at: interaction, arena: arena)
                case .scaledTexture(let value): return value.evaluateFloat(at: interaction, arena: arena)
                }
        }

        func evaluate(at interaction: any Interaction, arena: TextureArena) -> any TextureEvaluation {
                return evaluateFloat(at: interaction, arena: arena)
        }
}
