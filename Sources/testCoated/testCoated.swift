import libgonzales

print("test")

let shadingFrame = ShadingFrame(normal: upNormal)
let bsdfFrame = BsdfFrame(geometricNormal: upNormal, shadingFrame: shadingFrame)
let coated = CoatedDiffuseBsdf(
        reflectance: red,
        refractiveIndex: 1.0,
        roughness: (0.0, 0.0),
        remapRoughness: false,
        bsdfFrame: bsdfFrame)
let wo = Vector(x: 0.0, y: 0.1, z: 1.0)
let wi = Vector(x: 0.0, y: 0.0, z: 1.0)
let spectrum = coated.evaluateWorld(wo: wo, wi: wi)
print(spectrum)


