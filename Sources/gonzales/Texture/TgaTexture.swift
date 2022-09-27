import Foundation

final class TgaTexture: SpectrumTexture {

        init(width: Int, height: Int, channels: Int, data: [UInt8]) {
                self.width = width
                self.height = height
                self.channels = channels
                self.data = data
        }

        convenience init(path: String) throws {

                func get16Bit(from data: Data, at index: Int) -> UInt16 {
                        let a = data[index]
                        let b = data[index + 1]
                        return UInt16(a) + UInt16(b) * 256
                }

                guard let handle = FileHandle(forReadingAtPath: path) else {
                        throw RenderError.noFileHandle
                }
                let data = handle.readDataToEndOfFile()
                let width = Int(get16Bit(from: data, at: 12))
                let height = Int(get16Bit(from: data, at: 14))
                let bits = data[16]
                let descriptor = data[17]
                let rightToLeft = descriptor & 4
                let topToBottom = descriptor & 5
                let channels = Int(bits / 8)
                let length = UInt(width) * UInt(height) * UInt(channels)
                let end = 18 + length
                var imageData: [UInt8] = []
                imageData = Array(data[18..<end])
                if rightToLeft == 1 {
                        for y in 0..<height {
                                let left = y * width
                                for x in 0..<width / 2 {
                                        let from = left + x
                                        let to = left + width - x
                                        for c in 0..<channels {
                                                let from = from * channels + c
                                                let to = to * channels + c
                                                imageData.swapAt(from, to)
                                        }

                                }
                        }
                }
                if topToBottom == 1 {
                        for y in 0..<height / 2 {
                                let from = y
                                let to = height - y
                                for x in 0..<width {
                                        let from = from + x
                                        let to = to + x
                                        for c in 0..<channels {
                                                let from = from + c
                                                let to = to + c
                                                imageData.swapAt(from, to)
                                        }
                                }
                        }
                }
                self.init(width: width, height: height, channels: channels, data: imageData)
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
                        let b = data[index + 0]
                        let g = data[index + 1]
                        let r = data[index + 2]
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
                        let y = Int(uv[0] * FloatX(width))
                        let x = Int(uv[1] * FloatX(height))
                        return (x, y)
                }

                let (x, y) = getImageCoordinates(from: interaction.uv)
                let index = x * width * channels + y * channels
                guard index < width * height * channels else { return black }
                let ints = getRGB(from: index)
                let srgb = getSpectrum(from: ints)
                return inverseGamma(srgb)
        }

        let width: Int
        let height: Int
        let channels: Int
        let data: [UInt8]
}
