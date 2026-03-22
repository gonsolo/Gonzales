import Foundation

final class MixMaterial: Sendable {

        init(
                material1: Material,
                material2: Material,
                amount: FloatTexture
        ) {
                self.material1 = material1
                self.material2 = material2
                self.amount = amount
        }

        func getBsdf(interaction: SurfaceInteraction) throws -> MixBsdf {
                let amountAtInteraction = amount.evaluateFloat(at: interaction)
                let bsdfFrame = BsdfFrame(interaction: interaction)

                let bsdf1 = try material1.getBsdf(interaction: interaction)
                let bsdf2 = try material2.getBsdf(interaction: interaction)

                let mixBsdf = MixBsdf(
                        bsdf1: bsdf1,
                        bsdf2: bsdf2,
                        amount: amountAtInteraction,
                        bsdfFrame: bsdfFrame
                )
                return mixBsdf
        }

        let material1: Material
        let material2: Material
        let amount: FloatTexture
}

extension MixMaterial {

        static func create(parameters: ParameterDictionary, textures: [String: Texture], namedMaterials: [String: UninstancedMaterial], state: State) throws -> MixMaterial {
                
                guard let materials = try? parameters.findStrings(name: "materials"), materials.count == 2 else {
                        print("Warning: mix material expects exactly 2 materials.")
                        let defaultMat = try state.createMaterial(parameters: ParameterDictionary(), currentMaterial: nil, currentNamedMaterial: "defaultMaterial", textures: textures)
                        let constantAmount = ConstantTexture<Real>(value: 0.5)
                        let amountTexture = FloatTexture.constantTexture(constantAmount)
                        return MixMaterial(material1: defaultMat, material2: defaultMat, amount: amountTexture)
                }

                let material1Name = materials[0]
                let material2Name = materials[1]

                let material1 = try state.createMaterial(parameters: ParameterDictionary(), currentMaterial: nil, currentNamedMaterial: material1Name, textures: textures)
                let material2 = try state.createMaterial(parameters: ParameterDictionary(), currentMaterial: nil, currentNamedMaterial: material2Name, textures: textures)

                var amount: FloatTexture
                let textureName = (try? parameters.findTexture(name: "amount")) ?? ""
                if textureName != "", let tex = textures[textureName], case .floatTexture(let ftex) = tex {
                        amount = ftex
                } else if let scalarAmount = try? parameters.findOneReal(called: "amount", else: 0.5) {
                        amount = FloatTexture.constantTexture(ConstantTexture<Real>(value: scalarAmount))
                } else {
                        amount = FloatTexture.constantTexture(ConstantTexture<Real>(value: 0.5))
                }

                return MixMaterial(
                        material1: material1,
                        material2: material2,
                        amount: amount
                )
        }
}
