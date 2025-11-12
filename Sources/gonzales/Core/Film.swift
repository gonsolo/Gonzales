struct Film {

        enum FilmError: Error {
                case unknownFileType(name: String)
        }

        init(name: String, resolution: Point2i, fileName: String, filter: any Filter, crop: Bounds2f) {

                func upperPoint2i(_ point: Point2f) -> Point2i {
                        return Point2i(x: Int(point.x.rounded(.up)), y: Int(point.y.rounded(.up)))
                }

                self.name = name
                self.resolution = resolution
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

        private func add(
                samples: [Sample], image: inout Image, albedoImage: inout Image, normalImage: inout Image
        ) {
                for sample in samples {
                        add(
                                sample: sample,
                                image: &image,
                                albedoImage: &albedoImage,
                                normalImage: &normalImage)
                }
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

        func add(
                sample: Sample, image: inout Image, albedoImage: inout Image, normalImage: inout Image
        ) {
                if isWithin(location: sample.pixel, resolution: image.getResolution()) {
                        image.addPixel(
                                withColor: sample.light, withWeight: sample.weight, atLocation: sample.pixel)
                        albedoImage.addPixel(
                                withColor: sample.albedo, withWeight: sample.weight, atLocation: sample.pixel)
                        normalImage.addPixel(
                                withColor: RgbSpectrum(from: sample.normal), withWeight: sample.weight,
                                atLocation: sample.pixel)
                }
        }

        func getResolution() -> Point2i {
                return resolution
        }

        private let name: String
        private let fileName: String
        let filter: any Filter
        let resolution: Point2i
        var crop: Bounds2i
}
