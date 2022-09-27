import exr

final class ExrTexture: SpectrumTexture {

        init(width: Int, height: Int, channels: Int, data: [Float]) {
                self.width = width
                self.height = height
                self.channels = channels
                self.data = data
        }

        convenience init(path: String) {
                let widthPointer = UnsafeMutablePointer<Int32>.allocate(capacity: 1)
                widthPointer.initialize(repeating: 0, count: 1)
                let heightPointer = UnsafeMutablePointer<Int32>.allocate(capacity: 1)
                heightPointer.initialize(repeating: 0, count: 1)
                let pointer: UnsafeMutablePointer<Float> = readRgba(
                        path, widthPointer, heightPointer)
                let width = Int(widthPointer.pointee)
                let height = Int(heightPointer.pointee)
                let length: Int = width * height * 4
                let buffer = UnsafeBufferPointer(start: pointer, count: length)
                let array = Array(buffer)
                self.init(width: width, height: height, channels: 4, data: array)
        }

        func getRGB(from index: Int) -> (FloatX, FloatX, FloatX) {
                let r = FloatX(data[index + 0])
                let g = FloatX(data[index + 1])
                let b = FloatX(data[index + 2])
                return (r, g, b)
        }

        func evaluateSpectrum(at interaction: Interaction) -> Spectrum {

                func getImageCoordinates(from uv: Point2F) -> (Int, Int) {
                        var uv = uv
                        uv[0].formTruncatingRemainder(dividingBy: 1)
                        uv[1].formTruncatingRemainder(dividingBy: 1)
                        if uv[0] < 0 { uv[0] = 1 + uv[0] }
                        if uv[1] < 0 { uv[1] = 1 + uv[1] }
                        uv[1] = 1 - uv[1]
                        let y = Int(uv[0] * FloatX(width))
                        let x = Int(uv[1] * FloatX(height))
                        return (x, y)
                }

                let (x, y) = getImageCoordinates(from: interaction.uv)
                let index = x * width * channels + y * channels
                guard index < width * height * channels else { return black }
                let rgb = getRGB(from: index)
                let spectrum = Spectrum(r: rgb.0, g: rgb.1, b: rgb.2)
                return spectrum
        }

        let width: Int
        let height: Int
        let channels: Int
        let data: [Float]
}
