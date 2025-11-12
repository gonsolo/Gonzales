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

        func writeImages(samples: [Sample]) async throws {
                let processAndWrite:
                        @Sendable (_ name: String, _ fileName: String, _ isAlbedoOrNormal: Bool) async throws
                                -> Void =
                                { name, fileName, isAlbedoOrNormal in

                                        var image = Image(resolution: self.resolution)

                                        if isAlbedoOrNormal {
                                                if name.contains("albedo") {
                                                        self.add(samples: samples, albedoImage: &image)
                                                } else if name.contains("normal") {
                                                        self.add(samples: samples, normalImage: &image)
                                                } else {
                                                        self.add(samples: samples, image: &image)
                                                }
                                        } else {
                                                self.add(samples: samples, image: &image)
                                        }

                                        try image.normalize()
                                        let imageWriter = try self.chooseWriter(name: fileName)
                                        try imageWriter.write(
                                                fileName: fileName, crop: self.crop, image: image)
                                }

                try await withThrowingTaskGroup(of: Void.self) { group in

                        group.addTask {
                                try await processAndWrite(self.name, self.name, false)
                        }

                        group.addTask {
                                try await processAndWrite("albedo", "albedo.exr", true)
                        }

                        group.addTask {
                                try await processAndWrite("normal", "normal.exr", true)
                        }

                        try await group.waitForAll()
                }
        }
        private func add(samples: [Sample], image: inout Image) {
                for sample in samples {
                        add(sample: sample, image: &image)
                }
        }

        private func add(samples: [Sample], albedoImage: inout Image) {
                for sample in samples {
                        add(sample: sample, albedoImage: &albedoImage)
                }
        }

        private func add(samples: [Sample], normalImage: inout Image) {
                for sample in samples {
                        add(sample: sample, normalImage: &normalImage)
                }
        }

        private func isWithin<Point: GetIntXY>(location: Point, resolution: Point) -> Bool {
                return location.x >= 0 && location.y >= 0 && location.x < resolution.x
                        && location.y < resolution.y
        }

        private func isWithin<Point: GetFloatXY, Vector: GetFloatXY>(location: Point, support: Vector) -> Bool
        {
                return abs(location.x) < support.x && abs(location.y) < support.y
        }

        func add(sample: Sample, image: inout Image) {
                if isWithin(location: sample.pixel, resolution: image.getResolution()) {
                        image.addPixel(
                                withColor: sample.light, withWeight: sample.weight, atLocation: sample.pixel)
                }
        }

        func add(sample: Sample, albedoImage: inout Image) {
                if isWithin(location: sample.pixel, resolution: albedoImage.getResolution()) {
                        albedoImage.addPixel(
                                withColor: sample.albedo, withWeight: sample.weight, atLocation: sample.pixel)
                }
        }

        func add(sample: Sample, normalImage: inout Image) {
                if isWithin(location: sample.pixel, resolution: normalImage.getResolution()) {
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
