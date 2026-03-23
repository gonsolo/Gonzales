/// A collection of parameters.
///
/// Every parameter has a type an a value (or a collection of values).

typealias ParameterDictionary = [String: any Parameter]

extension ParameterDictionary {

        func findSpectrum(name: String, else spectrum: (any Spectrum)? = nil)
                throws -> (any Spectrum)? {
                let spectra = try findSpectra(name: name)
                if spectra.isEmpty {
                        return spectrum
                } else {
                        return spectra[0]
                }
        }

        func findOnePoint(name: String, else preset: Point) throws -> Point {
                let points = try findPoints(name: name)
                if points.isEmpty {
                        return preset
                } else {
                        return points[0]
                }
        }

        func findString(called name: String) throws -> String? {
                let strings = try findStrings(name: name)
                if strings.isEmpty {
                        return nil
                } else {
                        return strings[0]
                }
        }

        func findTexture(name: String) throws -> String {
                return try findString(called: name) ?? ""
        }

        func findRgbSpectrumTexture(
                name: String,
                textures: [String: Texture],
                arena: inout TextureArena,
                else spectrum: RgbSpectrum = RgbSpectrum(intensity: 1)
        )
                throws -> Texture {
                let textureName = try findTexture(name: name)

                if textureName != "" {
                        guard let texture = textures[textureName] else {
                                print(
                                        "Warning: Could not find texture \"\(textureName)\", using default spectrum."
                                )
                                let constantTexture = ConstantTexture(value: spectrum)
                                let index = arena.appendRgb(
                                        RgbSpectrumTexture.constantTexture(constantTexture))
                                return Texture.rgbSpectrumTexture(index)
                        }
                        switch texture {
                        case .floatTexture:
                                print("Warning: Could not find texture \(textureName)")
                                let constantTexture = ConstantTexture<RgbSpectrum>(value: red)
                                let index = arena.appendRgb(
                                        RgbSpectrumTexture.constantTexture(constantTexture))
                                return Texture.rgbSpectrumTexture(index)
                        case .rgbSpectrumTexture:
                                return texture
                        }
                } else {
                        guard let spectrum = try findSpectrum(name: name, else: spectrum) else {
                                throw RenderError.unimplemented(
                                        function: #function, file: #filePath, line: #line,
                                        message: "Missing spectrum")
                        }
                        guard let rgbSpectrum = spectrum as? RgbSpectrum else {
                                throw RenderError.unimplemented(
                                        function: #function, file: #filePath, line: #line,
                                        message: "Expected RgbSpectrum")
                        }
                        let constantTexture = ConstantTexture(value: rgbSpectrum)
                        let index = arena.appendRgb(RgbSpectrumTexture.constantTexture(constantTexture))
                        return Texture.rgbSpectrumTexture(index)
                }
        }

        func findRealTexture(
                name: String, textures: [String: Texture], arena: inout TextureArena,
                else value: Real = 1.0
        ) throws -> Texture {
                let textureName = try findTexture(name: name)
                if textureName != "" {
                        guard let texture = textures[textureName] else {
                                print(
                                        "Warning: No named texture \"\(textureName)\" found, using default value."
                                )
                                let index = arena.appendFloat(
                                        FloatTexture.constantTexture(ConstantTexture<Real>(value: value)))
                                return Texture.floatTexture(index)
                        }
                        switch texture {
                        case .floatTexture:
                                return texture
                        case .rgbSpectrumTexture:
                                print("No named float texture \"\(textureName)\"")
                                throw RenderError.unimplemented(
                                        function: #function, file: #filePath, line: #line,
                                        message: "No named float texture \"\(textureName)\"")
                        }
                } else {
                        let value = try findOneReal(called: name, else: value)
                        let index = arena.appendFloat(
                                FloatTexture.constantTexture(ConstantTexture<Real>(value: value)))
                        return Texture.floatTexture(index)
                }
        }

        func findOneBool(called name: String, else preset: Bool) throws -> Bool {
                let bools = try findBools(name: name)
                if bools.isEmpty {
                        return preset
                } else {
                        return bools[0]
                }
        }

        func findOneRealOptional(called name: String) throws -> Real? {
                let floats = try findReals(called: name)
                if floats.isEmpty {
                        return nil
                } else {
                        return floats[0]
                }
        }

        func findOneReal(called name: String, else preset: Real) throws -> Real {
                let floats = try findReals(called: name)
                if floats.isEmpty {
                        return preset
                } else {
                        return floats[0]
                }
        }

        func findOneInt(called name: String, else preset: Int) throws -> Int {
                let ints = try findInts(name: name)
                if ints.isEmpty {
                        return preset
                } else {
                        return ints[0]
                }
        }

        func findReals(called name: String) throws -> [Real] {
                guard let floats = self[name] as? [Real] else {
                        return []
                }
                return floats
        }

        func findPoints(name: String) throws -> [Point] {
                guard let points = self[name] as? [Point] else {
                        return []
                }
                return points
        }

        func findInts(name: String) throws -> [Int] {
                guard let ints = self[name] as? [Int] else {
                        return []
                }
                return ints
        }

        func findNormals(name: String) throws -> [Normal] {
                guard let normals = self[name] as? [Normal] else {
                        return []
                }
                return normals
        }

        func findSpectra(name: String) throws -> [any Spectrum] {
                guard let spectra = self[name] as? [any Spectrum] else {
                        return []
                }
                return spectra
        }

        func findStrings(name: String) throws -> [String] {
                guard let strings = self[name] as? [String] else {
                        return []
                }
                return strings
        }

        func findBools(name: String) throws -> [Bool] {
                guard let bools = self[name] as? [Bool] else {
                        return []
                }
                return bools
        }
}
