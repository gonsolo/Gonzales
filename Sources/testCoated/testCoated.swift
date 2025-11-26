import libgonzales

print("test")

let normal = Normal(x: 0, y: 0, z: 1)
let shadingFrame = ShadingFrame(
        tangent: Vector(x: 1, y: 0, z: 0),
        bitangent: Vector(x: 0, y: 1, z: 0),
        normal: normal)
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
let spectrum = coated.evaluateLocal(wo: wo, wi: wi)
print(spectrum)
