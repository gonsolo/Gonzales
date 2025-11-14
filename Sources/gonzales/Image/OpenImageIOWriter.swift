import openImageIOBridge

// Assume bridge functions are imported, e.g., via a bridging header:
// func openImageForTiledWriting(_: UnsafePointer<Int8>!, _: Int32, _: Int32, _: Int32, _: Int32, _: Int32) -> UnsafeMutableRawPointer!
// func writeTiledImageBody(_: UnsafeMutableRawPointer!, _: UnsafePointer<Float>!, _: Int32, _: Int32, _: Int32, _: Int32, _: Int32) -> Bool
// func closeImageOutput(_: UnsafeMutableRawPointer!)

actor OpenImageIOWriter {

        func write(fileName: String, crop: Bounds2i, image: Image, tileSize: (Int, Int)) throws {

                var buffer = [Float]()

                func write(pixel: Pixel, at index: Int) {
                        buffer[index + 0] = Float(pixel.light.red)
                        buffer[index + 1] = Float(pixel.light.green)
                        buffer[index + 2] = Float(pixel.light.blue)
                        buffer[index + 3] = 1
                }

                let resolution = crop.pMax - crop.pMin
                buffer = [Float](repeating: 0, count: resolution.x * resolution.y * 4)
                for y in crop.pMin.y..<crop.pMax.y {
                        for x in crop.pMin.x..<crop.pMax.x {
                                let location = Point2i(x: x, y: y)
                                let pixel = image.getPixel(atLocation: location)
                                let px = x - crop.pMin.x
                                let py = y - crop.pMin.y
                                let index = py * resolution.x * 4 + px * 4
                                write(pixel: pixel, at: index)
                        }
                }
                
                // --- HOISTED C++ LOGIC START ---
                
                let xres = Int32(resolution.x)
                let yres = Int32(resolution.y)
                let tileWidth = Int32(tileSize.0)
                let tileHeight = Int32(tileSize.1)
                let channels: Int32 = 4

                // 1. Open the file and get the ImageOutput pointer
                guard let outputPointer = openImageForTiledWriting(
                    fileName, xres, yres, tileWidth, tileHeight, channels) else {
                    // C function prints error on failure
                    return
                }

                // 2. Ensure the output pointer is closed and deleted when the function exits
                defer {
                    closeImageOutput(outputPointer)
                }

                // 3. Write the image body using the pointer
                buffer.withUnsafeBufferPointer { bufferPointer in
                    let success = writeTiledImageBody(
                        outputPointer,
                        bufferPointer.baseAddress,
                        xres, yres, tileWidth, tileHeight, channels)
                    
                    if !success {
                        // C function prints error on tile write failure
                    }
                }
                
                // --- HOISTED C++ LOGIC END ---
        }

        func write(fileName: String, image: Image, tileSize: (Int, Int)) async throws {
                let crop = Bounds2i(
                        pMin: Point2i(x: 0, y: 0),
                        pMax: image.getResolution())
                try write(fileName: fileName, crop: crop, image: image, tileSize: tileSize)
        }

}
