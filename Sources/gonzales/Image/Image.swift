struct Image {

        init(resolution: Point2i) {
                fullResolution = resolution
                pixels = Array(repeating: Pixel(), count: resolution.x * resolution.y)
        }

        func getPixel(atLocation location: Point2i) -> Pixel {
                return pixels[getIndexFor(location: location)]
        }

        private func getIndexFor(location: Point2i) -> Int {
                return location.y * fullResolution.x + location.x

        }

        mutating func normalize() throws {
                pixels = try pixels.map { return try $0.normalized() }
        }

        mutating func addPixel(
                withColor color: RgbSpectrum,
                withWeight weight: FloatX,
                atLocation location: Point2i
        ) {
                let index = location.y * fullResolution.x + location.x
                if index >= 0 && index < pixels.count {
                        pixels[index] = pixels[index] + Pixel(light: color, weight: weight)
                }
        }

        func getResolution() -> Point2i {
                return fullResolution
        }

        var fullResolution: Point2i
        var pixels: [Pixel]
}
