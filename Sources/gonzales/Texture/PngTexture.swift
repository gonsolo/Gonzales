import Foundation
import PNG

enum PngTextureError: Error {
        case decompress
}

final class PngTexture: SpectrumTexture {

        init(width: Int, height: Int, channels: Int, data: [UInt8]) {
                self.width = width
                self.height = height
                self.channels = channels
                self.data = data
        }

        convenience init(path: String) throws {
                guard let image: PNG.Data.Rectangular = try .decompress(path: path) else {
                        throw PngTextureError.decompress
                }
                let pixels: [PNG.RGBA<UInt8>] = image.unpack(as: PNG.RGBA<UInt8>.self)
                let (x:width, y:height) = image.size

                let channels = 4
                var data: [UInt8] = []
                for pixel in pixels {
                        data.append(pixel.r)
                        data.append(pixel.g)
                        data.append(pixel.b)
                        data.append(pixel.a)
                }
                self.init(width: width, height: height, channels: channels, data: data)
        }

        func inverseGamma(_ x: FloatX) -> FloatX {
                if x <= 0.04045 {
                        return x / 12.92
                } else {
                        return pow((x + 0.055) / 1.055, 2.4)
                }
        }

        func inverseGamma(_ s: Spectrum) -> Spectrum {
                return Spectrum(r: inverseGamma(s.r), g: inverseGamma(s.g), b: inverseGamma(s.b))
        }

        func evaluateSpectrum(at interaction: Interaction) -> Spectrum {

                func getRGB(from index: Int) -> (UInt8, UInt8, UInt8) {
                        let r = data[index + 0]
                        let g = data[index + 1]
                        let b = data[index + 2]
                        return (r, g, b)
                }

                func getSpectrum(from ints: (UInt8, UInt8, UInt8)) -> Spectrum {

                        func toFloatX(_ x: UInt8) -> FloatX { return FloatX(x) / 255 }

                        let r = toFloatX(ints.0)
                        let g = toFloatX(ints.1)
                        let b = toFloatX(ints.2)
                        return Spectrum(r: r, g: g, b: b)
                }

                func getImageCoordinates(from uv: Point2F) -> (Int, Int) {
                        var uv = uv
                        uv[0].formTruncatingRemainder(dividingBy: 1)
                        uv[1].formTruncatingRemainder(dividingBy: 1)
                        if uv[0] < 0 { uv[0] = 1 + uv[0] }
                        if uv[1] < 0 { uv[1] = 1 + uv[1] }
                        let x = Int(uv[0] * FloatX(width))
                        uv[1] = 1.0 - uv[1]
                        let y = Int(uv[1] * FloatX(height))
                        return (x, y)
                }

                let (x, y) = getImageCoordinates(from: interaction.uv)
                let index = y * width * channels + x * channels
                guard index < width * height * channels else { return black }
                let ints = getRGB(from: index)
                let srgb = getSpectrum(from: ints)
                return srgb
                //return inverseGamma(srgb)
        }

        let width: Int
        let height: Int
        let channels: Int
        let data: [UInt8]
}
