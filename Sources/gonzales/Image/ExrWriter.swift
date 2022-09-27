import exr

final class ExrWriter: ImageWriter {

        func write(fileName: String, crop: Bounds2i, image: Image) throws {

                func write(pixel: Pixel, at index: Int) {
                        buffer[index + 0] = Float(pixel.light.r)
                        buffer[index + 1] = Float(pixel.light.g)
                        buffer[index + 2] = Float(pixel.light.b)
                        buffer[index + 3] = 1
                }

                let resolution = crop.pMax - crop.pMin
                buffer = [Float](repeating: 0, count: resolution.x * resolution.y * 4)
                for y in crop.pMin.y..<crop.pMax.y {
                        for x in crop.pMin.x..<crop.pMax.x {
                                let location = Point2I(x: x, y: y)
                                let pixel = image.getPixel(atLocation: location)
                                let px = x - crop.pMin.x
                                let py = y - crop.pMin.y
                                let index = py * resolution.x * 4 + px * 4
                                write(pixel: pixel, at: index)
                        }
                }
                writeRgba(fileName, buffer, Int32(resolution.x), Int32(resolution.y))
        }

        var buffer = [Float]()
}
