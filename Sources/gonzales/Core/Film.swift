actor Film: Sendable {

        enum FilmError: Error {
                case unknownFileType(name: String)
        }

        init(name: String, resolution: Point2i, fileName: String, filter: any Filter, crop: Bounds2f) {

                func upperPoint2i(_ p: Point2f) -> Point2i {
                        return Point2i(x: Int(p.x.rounded(.up)), y: Int(p.y.rounded(.up)))
                }

                self.name = name
                self.image = Image(resolution: resolution)
                self.albedoImage = Image(resolution: resolution)
                self.normalImage = Image(resolution: resolution)
                self.fileName = fileName
                self.filter = filter
                self.crop = Bounds2i(
                        pMin: upperPoint2i(resolution * crop.pMin),
                        pMax: upperPoint2i(resolution * crop.pMax))
        }

        func getSampleBounds() -> Bounds2i {
                return crop
        }

        private func chooseWriter(name: String) throws -> ImageWriter {

                if name.hasSuffix(".exr") {
                        return ImageWriter.openImageIO(OpenImageIOWriter())
                } else if name.hasSuffix(".ascii") {
                        return ImageWriter.ascii(AsciiWriter())
                } else {
                        throw FilmError.unknownFileType(name: name)
                }
        }

        func writeImages() async throws {
                try await writeImage(name: fileName, image: image)
                try await writeImage(name: "albedo.exr", image: albedoImage)
                try await writeImage(name: "normal.exr", image: normalImage)
        }

        func writeImage(name: String, image: Image) async throws {
                try await image.normalize()
                let imageWriter = try chooseWriter(name: name)
                try await imageWriter.write(fileName: name, crop: crop, image: image)
        }

        func add(samples: [Sample]) async {
                for sample in samples {
                        await add(
                                value: sample.light,
                                weight: sample.weight,
                                location: sample.location,
                                image: image)
                        await add(
                                value: sample.albedo,
                                weight: sample.weight,
                                location: sample.location,
                                image: albedoImage)
                        await add(
                                value: RgbSpectrum(from: sample.normal),
                                weight: sample.weight,
                                location: sample.location,
                                image: normalImage)
                }
        }

        private func getRasterBound(from location: FloatX, delta: FloatX) -> Int {
                return Int((location + delta).rounded(.towardZero))
        }

        private func getRasterBounds(from location: FloatX, delta: FloatX) -> (Int, Int) {
                let min = getRasterBound(from: location, delta: -delta)
                let max = getRasterBound(from: location, delta: +delta)
                return (min, max)
        }

        private func generateBound(location: Point2f, radius: Vector2F) -> Bounds2i {
                let (xmin, xmax) = getRasterBounds(from: location.x, delta: radius.x)
                let (ymin, ymax) = getRasterBounds(from: location.y, delta: radius.y)
                let pMin = Point2i(x: xmin, y: ymin)
                let pMax = Point2i(x: xmax, y: ymax)
                return Bounds2i(pMin: pMin, pMax: pMax)
        }

        private func isWithin<Point: GetIntXY>(location: Point, resolution: Point)
                -> Bool
        {
                return location.x >= 0 && location.y >= 0 && location.x < resolution.x
                        && location.y < resolution.y
        }

        private func isWithin<Point: GetFloatXY, Vector: GetFloatXY>(location: Point, support: Vector)
                -> Bool
        {
                return abs(location.x) < support.x && abs(location.y) < support.y
        }

        func filterAndWrite(
                sample: Point2f,
                pixel: Point2i,
                image: Image,
                value: RgbSpectrum,
                weight: FloatX
        ) async {
                if isWithin(location: pixel, resolution: await image.getResolution()) {
                        let pixelCenter = Point2f(
                                x: FloatX(pixel.x) + 0.5,
                                y: FloatX(pixel.y) + 0.5)
                        let relativeLocation = Point2f(from: sample - pixelCenter)
                        if isWithin(location: relativeLocation, support: filter.support) {
                                let color = filter.evaluate(atLocation: relativeLocation) * value
                                let weight = filter.evaluate(atLocation: relativeLocation) * weight
                                await image.addPixel(
                                        withColor: color,
                                        withWeight: weight,
                                        atLocation: pixel)
                        }
                }
        }

        private func add(value: RgbSpectrum, weight: FloatX, location: Point2f, image: Image) async {
                let bound = generateBound(location: location, radius: filter.support)
                for x in bound.pMin.x...bound.pMax.x {
                        for y in bound.pMin.y...bound.pMax.y {
                                let pixelLocation = Point2i(x: x, y: y)
                                await filterAndWrite(
                                        sample: location,
                                        pixel: pixelLocation,
                                        image: image,
                                        value: value,
                                        weight: weight)
                        }
                }
        }

        func getResolution() async -> Point2i {
                return await image.getResolution()
        }

        private let name: String
        private let fileName: String
        let filter: any Filter
        var image: Image
        var albedoImage: Image
        var normalImage: Image
        var crop: Bounds2i
}
