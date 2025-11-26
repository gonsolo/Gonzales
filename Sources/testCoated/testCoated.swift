import libgonzales

print("test")

let shadingFrame = ShadingFrame(normal: upNormal)
let bsdfFrame = BsdfFrame(geometricNormal: upNormal, shadingFrame: shadingFrame)

let alpha: (FloatX, FloatX) = (0.001, 0.001)
let distribution = TrowbridgeReitzDistribution(alpha: alpha)
let dielectric = DielectricBsdf(distribution: distribution, refractiveIndex: 1.0, bsdfFrame: bsdfFrame)
let diffuse = DiffuseBsdf(reflectance: red, bsdfFrame: bsdfFrame)

let coated = CoatedDiffuseBsdf(
        dielectric: dielectric,
        diffuse: diffuse,
        bsdfFrame: bsdfFrame)

//CoatedDiffuseBxDF coated(DielectricBxDF(sampledEta, distrib), DiffuseBxDF(r), thick, a, gg, maxDepth, nSamples);

let wo = Vector(x: 0.0, y: 0.1, z: 1.0)
let wi = Vector(x: 0.0, y: 0.0, z: 1.0)
let spectrum = coated.evaluateWorld(wo: wo, wi: wi)
print(spectrum)
