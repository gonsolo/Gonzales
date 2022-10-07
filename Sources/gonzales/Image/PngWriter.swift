import PNG

final class PngWriter: ImageWriter {

        func clamp(value: FloatX, minValue: FloatX, maxValue: FloatX) -> FloatX {
                if value.isNaN { return maxValue }
                if value > maxValue {
                        return maxValue
                } else if value < minValue {
                        return minValue
                } else {
                        return value
                }
        }

        func write(fileName: String, crop: Bounds2i, image: Image) throws {
                let resolution = crop.pMax - crop.pMin
                var buffer = [UInt8](repeating: 0, count: resolution.x * resolution.y * 3)

                func doubleToInt(_ x: FloatX) -> UInt8 {
                        return UInt8(clamp(value: 255 * x, minValue: 0, maxValue: 255))
                }

                func write(value: FloatX, at index: Int) {
                        buffer[index] = doubleToInt(value)
                }

                func write(pixel: Pixel, at index: Int) {
                        write(value: pixel.light.r, at: index + 0)
                        write(value: pixel.light.g, at: index + 1)
                        write(value: pixel.light.b, at: index + 2)
                }

                for y in crop.pMin.y..<crop.pMax.y {
                        for x in crop.pMin.x..<crop.pMax.x {
                                let location = Point2I(x: x, y: y)
                                var pixel = image.getPixel(atLocation: location)
                                pixel.light = gammaLinearToSrgb(light: pixel.light)
                                let px = x - crop.pMin.x
                                let py = y - crop.pMin.y
                                let index = py * resolution.x * 3 + px * 3
                                write(pixel: pixel, at: index)
                        }
                }
                var pngBuffer: [PNG.RGBA<UInt8>] = []
                for y in 0..<resolution.y {
                        for x in 0..<resolution.x {
                                let index = y * resolution.y * 3 + x * 3
                                let rgba = PNG.RGBA<UInt8>(
                                        buffer[index], buffer[index + 1], buffer[index + 2], 255)
                                pngBuffer.append(rgba)
                        }
                }
                let image: PNG.Data.Rectangular = .init(
                        packing: pngBuffer,
                        size: (resolution.x, resolution.y),
                        layout: .init(format: .rgba8(palette: [], fill: nil)))
                try image.compress(path: fileName, level: 9)
        }
}
