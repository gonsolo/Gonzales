import libgonzales

print("test")

let red = RgbSpectrum(red: 1.0, green: 0.0, blue: 0.0)
let bsdfFrame = BsdfFrame(geometricNormal: Normal(x: 0.0, y: 0.0, z: 1.0), shadingFrame: ShadingFrame())
let coated = CoatedDiffuseBsdf(
                reflectance: red,
                refractiveIndex: 1.0,
                roughness: (1.0, 1.0),
                remapRoughness: false,
                bsdfFrame: bsdfFrame)
