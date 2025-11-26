import libgonzales

func test(shadingFrame: ShadingFrame) {
        print(shadingFrame)
        let worldVector = Vector(x: 1.0, y: 0.0, z: 0.0)
        print(worldVector)
        print(shadingFrame.worldToLocal(world: worldVector))
        let localVector = normalized(Vector(x: 0.3, y: 0.3, z: 0.3))
        print(localVector)
        print(shadingFrame.localToWorld(local: localVector))
        
        let bsdfFrame = BsdfFrame(geometricNormal: normal, shadingFrame: shadingFrame)
        
        let alpha: (FloatX, FloatX) = (0.001, 0.001)
        let distribution = TrowbridgeReitzDistribution(alpha: alpha)
        let dielectric = DielectricBsdf(distribution: distribution, refractiveIndex: 1.0, bsdfFrame: bsdfFrame)
        let diffuse = DiffuseBsdf(reflectance: red, bsdfFrame: bsdfFrame)
        
        let coated = CoatedDiffuseBsdf(
                dielectric: dielectric,
                diffuse: diffuse,
                bsdfFrame: bsdfFrame)
        
        let wo = Vector(x: 0.0, y: 0.0, z: 1.0)
        let wi = Vector(x: 0.0, y: 0.1, z: 1.0)
        let localSpectrum = coated.evaluateLocal(wo: wo, wi: wi)
        print("localSpectrum: ", localSpectrum)
        let worldSpectrum = coated.evaluateWorld(wo: wo, wi: wi)
        print("worldSpectrum: ", worldSpectrum)
}

let normal = Normal(x: 0, y: 0, z: 1)
let shadingFrame1 = ShadingFrame(
        tangent: Vector(x: 1, y: 0, z: 0),
        bitangent: Vector(x: 0, y: 1, z: 0),
        normal: normal)
test(shadingFrame: shadingFrame1)
let shadingFrame2 = ShadingFrame(
        tangent: Vector(x: 1, y: 0, z: 0),
        bitangent: Vector(x: 0, y: 1, z: 0))
test(shadingFrame: shadingFrame2)
let shadingFrame3 = ShadingFrame(
        normal: normalized(Normal(x: 1, y: 10, z: 1)))
test(shadingFrame: shadingFrame3)
