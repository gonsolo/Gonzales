final class Film {

        enum FilmError: Error {
                case unknownFileType(name: String)
        }

        init(name: String, resolution: Point2I, fileName: String, filter: Filter, crop: Bounds2f) {

                func upperPoint2i(_ p: Point2F) -> Point2I {
                        return Point2I(x: Int(p.x.rounded(.up)), y: Int(p.y.rounded(.up)))
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
                self.locker = Locker()
        }

        func getSampleBounds() -> Bounds2i {
                return crop
        }

        private func chooseWriter(name: String) throws -> ImageWriter {
                if name.hasSuffix(".exr") {
                        return ExrWriter()
                } else if name.hasSuffix(".ascii") {
                        return AsciiWriter()
                } else {
                        throw FilmError.unknownFileType(name: name)
                }
        }

        func writeImages() throws {
                try writeImage(name: fileName, image: &image)
                try writeImage(name: "albedo.exr", image: &albedoImage)
                try writeImage(name: "normal.exr", image: &normalImage)
        }

        func writeImage(name: String, image: inout Image) throws {
                try image.normalize()
                let imageWriter = try chooseWriter(name: name)
                try imageWriter.write(fileName: name, crop: crop, image: image)
        }

        func add(samples: [Sample]) {
                locker.locked {
                for sample in samples {
                        add(
                                value: sample.light,
                                weight: sample.weight,
                                location: sample.location,
                                image: &image)
                        //add(
                        //        value: sample.albedo,
                        //        weight: sample.weight,
                        //        location: sample.location,
                        //        image: &albedoImage)
                        //add(
                        //        value: RGBSpectrum(from: sample.normal),
                        //        weight: sample.weight,
                        //        location: sample.location,
                        //        image: &normalImage)
                }
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

        private func generateBound(location: Point2F, radius: Vector2F) -> Bounds2i {
                let (xmin, xmax) = getRasterBounds(from: location.x, delta: radius.x)
                let (ymin, ymax) = getRasterBounds(from: location.y, delta: radius.y)
                let pMin = Point2I(x: xmin, y: ymin)
                let pMax = Point2I(x: xmax, y: ymax)
                return Bounds2i(pMin: pMin, pMax: pMax)
        }

        private func isWithin(location: Point2I, resolution: Point2I) -> Bool {
                return location.x >= 0 && location.y >= 0 && location.x < resolution.x
                        && location.y < resolution.y
        }

        private func isWithin(location: Point2F, support: Vector2F) -> Bool {
                return abs(location.x) < support.x && abs(location.y) < support.y
        }

        func filterAndWrite(
                sample: Point2F,
                pixel: Point2I,
                image: inout Image,
                value: RGBSpectrum,
                weight: FloatX
        ) {
                if isWithin(location: pixel, resolution: image.fullResolution) {
                        let pixelCenter = Point2F(
                                x: FloatX(pixel.x) + 0.5,
                                y: FloatX(pixel.y) + 0.5)
                        let relativeLocation = Point2F(from: sample - pixelCenter)
                        if isWithin(location: relativeLocation, support: filter.support) {
                                let color = filter.evaluate(atLocation: relativeLocation) * value
                                let weight = filter.evaluate(atLocation: relativeLocation) * weight
                                image.addPixel(
                                        withColor: color,
                                        withWeight: weight,
                                        atLocation: pixel)
                        }
                }
        }

        private func add(value: RGBSpectrum, weight: FloatX, location: Point2F, image: inout Image)
        {
                let bound = generateBound(location: location, radius: filter.support)
                for x in bound.pMin.x...bound.pMax.x {
                        for y in bound.pMin.y...bound.pMax.y {
                                let pixelLocation = Point2I(x: x, y: y)
                                filterAndWrite(
                                        sample: location,
                                        pixel: pixelLocation,
                                        image: &image,
                                        value: value,
                                        weight: weight)
                        }
                }
        }

        private var name: String
        private var fileName: String
        var filter: Filter
        var image: Image
        var albedoImage: Image
        var normalImage: Image
        var crop: Bounds2i
        let locker: Locker
}
