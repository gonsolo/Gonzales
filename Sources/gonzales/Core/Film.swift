struct Film {

        enum FilmError: Error {
                case unknownFileType(name: String)
        }

        init(name: String, resolution: Point2i, fileName: String, filter: any Filter, crop: Bounds2f) {

                func upperPoint2i(_ p: Point2f) -> Point2i {
                        return Point2i(x: Int(p.x.rounded(.up)), y: Int(p.y.rounded(.up)))
                }

                self.name = name
                self.resolution = resolution
                //self.image = Image(resolution: resolution)
                //self.albedoImage = Image(resolution: resolution)
                //self.normalImage = Image(resolution: resolution)
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

        func writeImages(samples: [Sample]) throws {

                var image = Image(resolution: resolution)
                var albedoImage = Image(resolution: resolution)
                var normalImage = Image(resolution: resolution)

                add(samples: samples, image: &image, albedoImage: &albedoImage, normalImage: &normalImage)

                try image.normalize()
                let imageWriter = try chooseWriter(name: name)
                try imageWriter.write(fileName: name, crop: crop, image: image)

                try albedoImage.normalize()
                let albedoImageWriter = try chooseWriter(name: "albedo.exr")
                try albedoImageWriter.write(fileName: "albedo.exr", crop: crop, image: image)

                try normalImage.normalize()
                let normalImageWriter = try chooseWriter(name: "normal.exr")
                try normalImageWriter.write(fileName: "normal.exr", crop: crop, image: image)
        }

        private func add(samples: [Sample], image: inout Image, albedoImage: inout Image, normalImage: inout Image) {
                for sample in samples {
                        add(
                                value: sample.light,
                                albedo: sample.albedo,
                                normal: RgbSpectrum(from: sample.normal),
                                weight: sample.weight,
                                location: sample.location,
                                image: &image,
                                albedoImage: &albedoImage,
                                normalImage: &normalImage)
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
                value: RgbSpectrum,
                albedo: RgbSpectrum,
                normal: RgbSpectrum,
                weight: FloatX,
                image: inout Image,
                albedoImage: inout Image,
                normalImage: inout Image
        ) {
                if isWithin(location: pixel, resolution: image.getResolution()) {
                        let pixelCenter = Point2f(
                                x: FloatX(pixel.x) + 0.5,
                                y: FloatX(pixel.y) + 0.5)
                        let relativeLocation = Point2f(from: sample - pixelCenter)
                        if isWithin(location: relativeLocation, support: filter.support) {
                                let color = filter.evaluate(atLocation: relativeLocation) * value
                                let albedoColor = filter.evaluate(atLocation: relativeLocation) * albedo
                                let normalColor = filter.evaluate(atLocation: relativeLocation) * normal
                                let weight = filter.evaluate(atLocation: relativeLocation) * weight
                                image.addPixel( withColor: color, withWeight: weight, atLocation: pixel)
                                albedoImage.addPixel( withColor: albedoColor, withWeight: weight, atLocation: pixel)
                                normalImage.addPixel( withColor: normalColor, withWeight: weight, atLocation: pixel)
                        }
                }
        }

        func add(value: RgbSpectrum, albedo: RgbSpectrum, normal: RgbSpectrum, weight: FloatX, location: Point2f,
                image: inout Image, albedoImage: inout Image, normalImage: inout Image) {
                let bound = generateBound(location: location, radius: filter.support)
                for x in bound.pMin.x...bound.pMax.x {
                        for y in bound.pMin.y...bound.pMax.y {
                                let pixelLocation = Point2i(x: x, y: y)
                                filterAndWrite(
                                        sample: location,
                                        pixel: pixelLocation,
                                        value: value,
                                        albedo: albedo,
                                        normal: normal,
                                        weight: weight,
                                        image: &image,
                                        albedoImage: &albedoImage,
                                        normalImage: &normalImage)
                        }
                }
        }

        func getResolution() -> Point2i {
                return resolution
        }

        private let name: String
        private let fileName: String
        let filter: any Filter
        //var image: Image
        //var albedoImage: Image
        //var normalImage: Image
        let resolution: Point2i
        var crop: Bounds2i
}
