import Foundation  // DispatchSemaphore

enum ImageError: Error {
        case lock
}

struct Image {

        init(resolution: Point2I) {
                fullResolution = resolution
                pixels = Array(repeating: Pixel(), count: resolution.x * resolution.y)
        }

        func getPixel(atLocation location: Point2I) -> Pixel {
                return pixels[getIndexFor(location: location)]
        }

        private func getIndexFor(location: Point2I) -> Int {
                return location.y * fullResolution.x + location.x

        }

        mutating func normalize() throws {
                pixels = try pixels.map { return try $0.normalized() }
        }

        mutating func addPixel(
                withColor color: RGBSpectrum,
                withWeight weight: FloatX,
                atLocation location: Point2I
        ) {
                let index = location.y * fullResolution.x + location.x
                if index >= 0 && index < pixels.count {
                        pixels[index] = pixels[index] + Pixel(light: color, weight: weight)
                }
        }

        var fullResolution: Point2I
        var pixels: [Pixel]
}
