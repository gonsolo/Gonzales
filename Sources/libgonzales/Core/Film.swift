struct Film {

        enum FilmError: Error {
                case unknownFileType(name: String)
        }

        init(
                name: String, resolution: Point2i, fileName: String, filter: any Filter, crop: Bounds2f,
                iso: Real = 100.0, maxComponentValue: Real = 0.0
        ) {

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
                self.iso = iso
                self.maxComponentValue = maxComponentValue
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

        func writeImages(samples: [Sample], tileSize: (Int, Int)) async throws {

                // 1. Build all three images from samples
                var beautyImage = Image(resolution: self.resolution)
                var albedoImage = Image(resolution: self.resolution)
                var normalImage = Image(resolution: self.resolution)

                self.add(samples: samples, image: &beautyImage)
                self.add(samples: samples, albedoImage: &albedoImage)
                self.add(samples: samples, normalImage: &normalImage)

                // 2. Normalize
                try beautyImage.normalize()
                try albedoImage.normalize()
                try normalImage.normalize()

                // 3. Denoise beauty using albedo + normal as auxiliary inputs
                Denoiser.denoise(beauty: &beautyImage, albedo: albedoImage, normal: normalImage)

                // 4. Write all images to disk (copy to let bindings for sendability)
                let finalBeauty = beautyImage
                let finalAlbedo = albedoImage
                let finalNormal = normalImage

                let writeImage: @Sendable (_ fileName: String, _ image: Image) async throws -> Void = {
                        fileName, image in
                        let imageWriter = try self.chooseWriter(name: fileName)
                        try await imageWriter.write(
                                fileName: fileName, crop: self.crop, image: image,
                                tileSize: tileSize)
                }

                try await withThrowingTaskGroup(of: Void.self) { group in
                        group.addTask { try await writeImage(self.name, finalBeauty) }
                        group.addTask { try await writeImage("albedo.exr", finalAlbedo) }
                        group.addTask { try await writeImage("normal.exr", finalNormal) }
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

        private func isWithin<Point: GetRealY, Vector: GetRealY>(location: Point, support: Vector) -> Bool {
                return abs(location.x) < support.x && abs(location.y) < support.y
        }

        func add(sample: Sample, image: inout Image) {
                if isWithin(location: sample.pixel, resolution: image.getResolution()) {
                        var scaledLight = sample.light * RgbSpectrum(intensity: iso / 100.0)
                        if maxComponentValue > 0 {
                                let maxVal = scaledLight.maxValue
                                if maxVal > maxComponentValue {
                                        scaledLight *= maxComponentValue / maxVal
                                }
                        }
                        image.addPixel(
                                withColor: scaledLight, withWeight: sample.weight, atLocation: sample.pixel)
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
        let iso: Real
        let maxComponentValue: Real
}
