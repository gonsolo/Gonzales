import Foundation  // pow

private func schlickWeight(_ cosTheta: FloatX) -> FloatX {
        let m = clamp(value: 1 - cosTheta, low: 0, high: 1)
        return m * m * m * m * m
}

private func fresnelSchlick(r0: Spectrum, cosTheta: FloatX) -> Spectrum {
        return lerp(with: Spectrum(intensity: schlickWeight(cosTheta)), between: r0, and: white)
}

struct DisneyBSSRDF: BSSRDF {

        init(reflectance: Spectrum, distance: Spectrum, interaction: Interaction, eta: FloatX) {
                // TODO
        }
}

final class DisneyDiffuse: BxDF {

        init(reflectance: Spectrum) {
                self.reflectance = reflectance
        }

        func evaluate(wo: Vector, wi: Vector) -> Spectrum {
                let weightOut = schlickWeight(absCosTheta(wo))
                let weightIn = schlickWeight(absCosTheta(wi))
                let result = reflectance * (1 - weightOut / 2) * (1 - weightIn / 2) / FloatX.pi
                return result
        }

        func albedo() -> Spectrum { return reflectance }

        var reflectance: Spectrum
}

final class DisneyFakeSubsurface: BxDF {

        init(reflectance: Spectrum, roughness: FloatX) {
                self.reflectance = reflectance
                self.roughness = roughness
        }

        func evaluate(wo: Vector, wi: Vector) -> Spectrum {
                var half = wi + wo
                guard !half.isZero else { return black }
                half.normalize()
                let cosThetaD = dot(wi, half)
                let cosThetaIn = absCosTheta(wi)
                let cosThetaOut = absCosTheta(wo)
                let fss90 = cosThetaD * cosThetaD * roughness
                let weightOut = schlickWeight(cosThetaOut)
                let weightIn = schlickWeight(cosThetaIn)
                let fss =
                        lerp(with: weightOut, between: 1, and: fss90)
                        * lerp(with: weightIn, between: 1, and: fss90)
                let preserveAlbedo: FloatX = 1.25
                let ss =
                        preserveAlbedo
                        * (fss * (FloatX(1) / (cosThetaOut + cosThetaIn) - FloatX(0.5))
                                + FloatX(0.5))
                let result = reflectance * ss / FloatX.pi
                return result
        }

        func albedo() -> Spectrum { return reflectance }

        var reflectance: Spectrum
        var roughness: FloatX
}

final class DisneyRetroReflection: BxDF {

        init(reflectance: Spectrum, roughness: FloatX) {
                self.reflectance = reflectance
                self.roughness = roughness
        }

        func evaluate(wo: Vector, wi: Vector) -> Spectrum {
                var half = wi + wo
                if half.isZero { return black }
                half.normalize()
                let cosThetaD = dot(wi, half)
                let weightOut = schlickWeight(absCosTheta(wo))
                let weightIn = schlickWeight(absCosTheta(wi))
                let rr = 2 * roughness * cosThetaD * cosThetaD
                let result =
                        reflectance * rr * (weightOut + weightIn + weightOut * weightIn * (rr - 1))
                        / FloatX.pi
                return result
        }

        func albedo() -> Spectrum { return reflectance }

        var reflectance: Spectrum
        var roughness: FloatX
}

final class DisneySheen: BxDF {

        init(reflectance: Spectrum) {
                self.reflectance = reflectance
        }

        func evaluate(wo: Vector, wi: Vector) -> Spectrum {
                var half = wi + wo
                if half.isZero { return black }
                half.normalize()
                let cosThetaD = dot(wi, half)
                let result = reflectance * schlickWeight(cosThetaD)
                return result
        }

        func albedo() -> Spectrum { return reflectance }

        var reflectance: Spectrum
}

final class DisneyMicrofacetDistribution: MicrofacetDistribution {

        init(alpha: (FloatX, FloatX)) {
                trowbridgeReitzDistribution = TrowbridgeReitzDistribution(alpha: alpha)
        }

        func differentialArea(withNormal half: Vector) -> FloatX {
                return trowbridgeReitzDistribution.differentialArea(withNormal: half)
        }

        func lambda(_ vector: Vector) -> FloatX {
                return trowbridgeReitzDistribution.lambda(vector)
        }

        func sampleHalfVector(wo: Vector, u: Point2F) -> Vector {
                return trowbridgeReitzDistribution.sampleHalfVector(wo: wo, u: u)
        }

        func visibleFraction(from wo: Vector, and wi: Vector) -> FloatX {
                return maskingShadowing(wo) * maskingShadowing(wi)
        }

        let trowbridgeReitzDistribution: TrowbridgeReitzDistribution
}

struct DisneyFresnel: Fresnel {

        func evaluate(cosTheta: FloatX) -> Spectrum {
                return lerp(
                        with: Spectrum(intensity: metallic),
                        between: Spectrum(
                                intensity: frDielectric(cosThetaI: cosTheta, etaI: 1, etaT: eta)),
                        and: fresnelSchlick(r0: r0, cosTheta: cosTheta))
        }

        let r0: Spectrum
        let metallic: FloatX
        let eta: FloatX
}

final class DisneyClearCoat: BxDF {

        init(weight: FloatX, gloss: FloatX) {
                self.weight = weight
                self.gloss = gloss
        }

        private func gtr1(cosTheta: FloatX, alpha: FloatX) -> FloatX {
                let alpha2 = alpha * alpha
                let cosTheta2 = cosTheta * cosTheta
                return (alpha2 - 1) / (FloatX.pi * log(alpha2) * (1 + (alpha2 - 1) * cosTheta2))
        }

        private func frSchlick(r0: FloatX, cosTheta: FloatX) -> FloatX {
                lerp(with: schlickWeight(cosTheta), between: r0, and: 1)
        }

        private func smithGGX(cosTheta: FloatX, alpha: FloatX) -> FloatX {
                let alpha2 = square(alpha)
                let cosTheta2 = square(cosTheta)
                return 1 / (cosTheta + (alpha2 + cosTheta2 - alpha2 * cosTheta2).squareRoot())
        }

        func evaluate(wo: Vector, wi: Vector) -> Spectrum {
                var half = wi + wo
                guard !half.isZero else { return black }
                half.normalize()
                let dr = gtr1(cosTheta: absCosTheta(half), alpha: gloss)
                let fr = frSchlick(r0: 0.04, cosTheta: dot(wo, half))
                let gr =
                        smithGGX(cosTheta: absCosTheta(wo), alpha: 0.25)
                        * smithGGX(cosTheta: absCosTheta(wi), alpha: 0.25)
                return Spectrum(intensity: weight * gr * fr * dr / 4)
        }

        func sample(wo: Vector, u: Point2F) -> (Spectrum, Vector, FloatX) {
                if wo.z == 0 {
                        return (black, nullVector, 0)
                }
                let alpha2 = gloss * gloss
                let cosTheta = (max(0, (1 - pow(alpha2, 1 - u[0])) / (1 - alpha2))).squareRoot()
                let sinTheta = (max(0, 1 - cosTheta * cosTheta)).squareRoot()
                let phi = 2 * FloatX.pi * u[1]
                var half = sphericalDirection(sinTheta: sinTheta, cosTheta: cosTheta, phi: phi)
                if !sameHemisphere(wo, half) { half = -half }
                let wi = reflect(vector: wo, by: half)
                // TODO
                return (black, wi, 0)
        }

        func probabilityDensity(wo: Vector, wi: Vector) -> FloatX {
                if !sameHemisphere(wo, wi) { return 0 }
                var half = wi + wo
                guard !half.isZero else { return 0 }
                half.normalize()
                let cosTheta = absCosTheta(half)
                let dr = gtr1(cosTheta: cosTheta, alpha: gloss)
                return dr * cosTheta / (4 * dot(wo, half))
        }

        func albedo() -> Spectrum {
                return Spectrum(intensity: weight * gloss)
        }

        var weight: FloatX
        var gloss: FloatX
}

final class Disney: Material {

        init(
                color: SpectrumTexture,
                scatterDistance: SpectrumTexture,
                anisotropic: FloatTexture,
                clearcoat: FloatTexture,
                clearcoatGloss: FloatTexture,
                diffTrans: FloatTexture,
                eta: FloatTexture,
                flatness: FloatTexture,
                metallic: FloatTexture,
                roughness: FloatTexture,
                sheen: FloatTexture,
                sheenTint: FloatTexture,
                specularTint: FloatTexture,
                specularTrans: FloatTexture,
                thin: Bool
        ) {
                self.color = color
                self.scatterDistance = scatterDistance
                self.anisotropic = anisotropic
                self.clearcoat = clearcoat
                self.clearcoatGloss = clearcoatGloss
                self.diffTrans = diffTrans
                self.eta = eta
                self.flatness = flatness
                self.metallic = metallic
                self.roughness = roughness
                self.specularTrans = specularTrans
                self.sheen = sheen
                self.sheenTint = sheenTint
                self.specularTint = specularTint
                self.thin = thin
        }

        private func schlickR0(from eta: FloatX) -> FloatX {
                return square(eta - 1) / square(eta + 1)
        }

        func computeScatteringFunctions(interaction: Interaction) -> (BSDF, BSSRDF?) {
                var bsdf = BSDF(interaction: interaction)
                var bssrdf: BSSRDF! = nil
                let metallic = self.metallic.evaluateFloat(at: interaction)
                let specularTrans = self.specularTrans.evaluateFloat(at: interaction)
                let diffuse = (1 - metallic) * (1 - specularTrans)
                let color = self.color.evaluateSpectrum(at: interaction)
                let roughness = self.roughness.evaluateFloat(at: interaction)
                let eta = self.eta.evaluateFloat(at: interaction)
                let luminance: FloatX = color.luminance
                var tintColor: Spectrum
                if luminance > 0 {
                        tintColor = color / luminance
                } else {
                        tintColor = white
                }
                let diffTrans = self.diffTrans.evaluateFloat(at: interaction) / 2
                let diffuseColor = diffuse * color

                if diffuse > 0 {
                        if thin {
                                let flatness = self.flatness.evaluateFloat(at: interaction)
                                let thinReflectance =
                                        diffuse * (1.0 - flatness) * (1.0 - diffTrans) * color
                                if !thinReflectance.isBlack {
                                        bsdf.add(bxdf: DisneyDiffuse(reflectance: thinReflectance))
                                }
                                let subsurfaceReflectance =
                                        diffuse * flatness * (1 - diffTrans) * color
                                if !subsurfaceReflectance.isBlack {
                                        bsdf.add(
                                                bxdf: DisneyFakeSubsurface(
                                                        reflectance: subsurfaceReflectance,
                                                        roughness: roughness))
                                }
                        } else {
                                let scatterDistance = self.scatterDistance.evaluateSpectrum(
                                        at: interaction)
                                if scatterDistance.isBlack {
                                        if !diffuseColor.isBlack {
                                                bsdf.add(
                                                        bxdf: DisneyDiffuse(
                                                                reflectance: diffuseColor))
                                        }
                                } else {
                                        bsdf.add(
                                                bxdf: SpecularTransmission(
                                                        t: white, etaA: 1, etaB: eta))
                                        if !diffuseColor.isBlack {
                                                bssrdf = DisneyBSSRDF(
                                                        reflectance: diffuseColor,
                                                        distance: scatterDistance,
                                                        interaction: interaction,
                                                        eta: eta)
                                        }
                                }
                        }
                        if !diffuseColor.isBlack {
                                bsdf.add(
                                        bxdf: DisneyRetroReflection(
                                                reflectance: diffuseColor, roughness: roughness))
                        }
                        let sheen = self.sheen.evaluateFloat(at: interaction)
                        if sheen > 0 {
                                let sheenTint = self.sheenTint.evaluateFloat(at: interaction)
                                let sheenColor = lerp(
                                        with: Spectrum(intensity: sheenTint), between: white,
                                        and: tintColor)
                                bsdf.add(
                                        bxdf: DisneySheen(reflectance: diffuse * sheen * sheenColor)
                                )
                        }
                }
                let aspect = (1 - anisotropic.evaluateFloat(at: interaction) * 0.9).squareRoot()
                let alpha = (
                        max(0.001, square(roughness) / aspect),
                        max(0.001, square(roughness) * aspect)
                )
                let distribution = DisneyMicrofacetDistribution(alpha: alpha)
                let specularTint = self.specularTint.evaluateFloat(at: interaction)
                let r0 = lerp(
                        with: Spectrum(intensity: metallic),
                        between: schlickR0(from: eta)
                                * lerp(
                                        with: Spectrum(intensity: specularTint),
                                        between: white,
                                        and: tintColor),
                        and: color)
                let fresnel = DisneyFresnel(r0: r0, metallic: metallic, eta: eta)
                bsdf.add(
                        bxdf: MicrofacetReflection(
                                reflectance: white,
                                distribution: distribution,
                                fresnel: fresnel))
                let clearcoat = self.clearcoat.evaluateFloat(at: interaction)
                let clearcoatGloss = self.clearcoatGloss.evaluateFloat(at: interaction)
                if clearcoat > 0 {
                        bsdf.add(
                                bxdf: DisneyClearCoat(
                                        weight: clearcoat,
                                        gloss: lerp(
                                                with: clearcoatGloss,
                                                between: 0.1,
                                                and: 0.001)))
                }
                if specularTrans > 0 {
                        let t = specularTrans * color.squareRoot()
                        if thin {
                                let scaledRoughness = (0.65 * eta - 0.35) * roughness
                                let squaredRoughness = max(0.001, square(scaledRoughness))
                                let alpha = (
                                        squaredRoughness / aspect,
                                        squaredRoughness * aspect
                                )
                                let scaledDistribution = TrowbridgeReitzDistribution(alpha: alpha)
                                let transmission = MicrofacetTransmission(
                                        t: t,
                                        distribution: scaledDistribution,
                                        eta: (1.0, eta))
                                bsdf.add(bxdf: transmission)
                        } else {
                                let transmission = MicrofacetTransmission(
                                        t: t,
                                        distribution: distribution,
                                        eta: (1.0, eta))
                                bsdf.add(bxdf: transmission)
                        }
                }
                if thin {
                        bsdf.add(bxdf: LambertianTransmission(reflectance: diffTrans * color))
                }
                return (bsdf, bssrdf)
        }

        let color: SpectrumTexture
        let scatterDistance: SpectrumTexture

        let anisotropic: FloatTexture
        let clearcoat: FloatTexture
        let clearcoatGloss: FloatTexture
        let diffTrans: FloatTexture
        let eta: FloatTexture
        let flatness: FloatTexture
        let metallic: FloatTexture
        let roughness: FloatTexture
        let sheen: FloatTexture
        let sheenTint: FloatTexture
        let specularTint: FloatTexture
        let specularTrans: FloatTexture

        let thin: Bool
}

func createDisney(parameters: ParameterDictionary) throws -> Disney {
        let color = try parameters.findSpectrumTexture(name: "color", else: gray)
        let scatterDistance = try parameters.findSpectrumTexture(
                name: "scatterdistance", else: black)
        let anisotropic = try parameters.findFloatXTexture(name: "anisotropic", else: 0)
        let clearcoat = try parameters.findFloatXTexture(name: "clearcoat", else: 0)
        let clearcoatGloss = try parameters.findFloatXTexture(name: "clearcoatgloss", else: 1)
        let diffTrans = try parameters.findFloatXTexture(name: "difftrans", else: 1)
        let eta = try parameters.findFloatXTexture(name: "eta", else: 1.5)
        let flatness = try parameters.findFloatXTexture(name: "flatness", else: 0)
        let metallic = try parameters.findFloatXTexture(name: "metallic", else: 0)
        let roughness = try parameters.findFloatXTexture(name: "roughness", else: 0.5)
        let sheen = try parameters.findFloatXTexture(name: "sheen", else: 0)
        let sheenTint = try parameters.findFloatXTexture(name: "sheentint", else: 0.5)
        let specularTint = try parameters.findFloatXTexture(name: "speculartint", else: 0)
        let specularTrans = try parameters.findFloatXTexture(name: "spectrans", else: 0)
        let thin = try parameters.findOneBool(called: "thin", else: false)
        return Disney(
                color: color,
                scatterDistance: scatterDistance,
                anisotropic: anisotropic,
                clearcoat: clearcoat,
                clearcoatGloss: clearcoatGloss,
                diffTrans: diffTrans,
                eta: eta,
                flatness: flatness,
                metallic: metallic,
                roughness: roughness,
                sheen: sheen,
                sheenTint: sheenTint,
                specularTint: specularTint,
                specularTrans: specularTrans,
                thin: thin)
}
