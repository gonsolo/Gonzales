import Foundation
import mojoKernel

struct Denoiser {

        static func denoise(beauty: inout Image, albedo: Image, normal: Image) {
                let resolution = beauty.getResolution()
                let width = resolution.x
                let height = resolution.y
                let pixelCount = width * height

                var colorBuffer = [Float](repeating: 0, count: pixelCount * 3)
                var albedoBuffer = [Float](repeating: 0, count: pixelCount * 3)
                var outputBuffer = [Float](repeating: 0, count: pixelCount * 3)

                for y in 0..<height {
                        for x in 0..<width {
                                let loc = Point2i(x: x, y: y)
                                let idx = (y * width + x) * 3
                                let bp = beauty.getPixel(atLocation: loc)
                                colorBuffer[idx + 0] = Float(bp.light.red)
                                colorBuffer[idx + 1] = Float(bp.light.green)
                                colorBuffer[idx + 2] = Float(bp.light.blue)
                                let ap = albedo.getPixel(atLocation: loc)
                                albedoBuffer[idx + 0] = Float(ap.light.red)
                                albedoBuffer[idx + 1] = Float(ap.light.green)
                                albedoBuffer[idx + 2] = Float(ap.light.blue)
                        }
                }

                print("Denoising...", terminator: "")
                let start = Date()

                colorBuffer.withUnsafeBufferPointer { colorPtr in
                        albedoBuffer.withUnsafeBufferPointer { albedoPtr in
                                outputBuffer.withUnsafeMutableBufferPointer { outPtr in
                                        mojo_denoise(
                                                colorPtr.baseAddress!,
                                                albedoPtr.baseAddress!,
                                                Int32(width), Int32(height),
                                                outPtr.baseAddress!,
                                                7, 5.0, 0.2)
                                }
                        }
                }

                let duration = Date().timeIntervalSince(start)
                print(String(format: "\rDenoising complete (%.3fs)", duration))

                for y in 0..<height {
                        for x in 0..<width {
                                let idx = (y * width + x) * 3
                                beauty.setPixel(
                                        color: RgbSpectrum(
                                                red: Real(outputBuffer[idx + 0]),
                                                green: Real(outputBuffer[idx + 1]),
                                                blue: Real(outputBuffer[idx + 2])),
                                        atLocation: Point2i(x: x, y: y))
                        }
                }
        }
}
