import openImageIOBridge
import Foundation // Needed for MemoryLayout

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
                
                // --- HOISTED TILING LOGIC START ---
                
                let xres = Int32(resolution.x)
                let yres = Int32(resolution.y)
                let tileWidth = Int32(tileSize.0)
                let tileHeight = Int32(tileSize.1)
                let channels: Int32 = 4

                // 1. Open the file
                guard let outputPointer = openImageForTiledWriting(
                    fileName, xres, yres, tileWidth, tileHeight, channels) else {
                    return
                }

                // 2. Ensure the output pointer is closed and deleted when the function exits
                defer {
                    closeImageOutput(outputPointer)
                }

                // 3. Stride Calculations (Calculated in Swift)
                let floatSize = Int64(MemoryLayout<Float>.size)
                let channelStride = floatSize // 4 bytes
                let xStride = Int64(channels) * channelStride // 16 bytes
                let yStride = Int64(xres) * xStride // xres * 16 bytes

                let nxtiles = (xres + tileWidth - 1) / tileWidth
                let nytiles = (yres + tileHeight - 1) / tileHeight
                
                // 4. Tiling Loop (Hoisted from C++)
                buffer.withUnsafeBufferPointer { bufferPointer in
                    guard let pixels = bufferPointer.baseAddress else { return }
                    
                    for ty in 0..<nytiles {
                        for tx in 0..<nxtiles {
                            let success = writeSingleTile(
                                outputPointer,
                                pixels,
                                xres, channels,
                                Int32(tx), Int32(ty), tileWidth, tileHeight,
                                channelStride, xStride, yStride // Stride values as Int64
                            )
                            
                            if !success {
                                return // Exit on first tile write failure
                            }
                        }
                    }
                }
                
                // --- HOISTED TILING LOGIC END ---
        }

        func write(fileName: String, image: Image, tileSize: (Int, Int)) async throws {
                let crop = Bounds2i(
                        pMin: Point2i(x: 0, y: 0),
                        pMax: image.getResolution())
                try write(fileName: fileName, crop: crop, image: image, tileSize: tileSize)
        }

}
