import openImageIOBridge
import Foundation // Needed for MemoryLayout

actor OpenImageIOWriter {

        func write(fileName: String, image: Image, tileSize: (Int, Int), crop: Bounds2i? = nil) async throws {

                // 1. Determine the final crop bounds
                let finalCrop: Bounds2i
                if let explicitCrop = crop {
                    finalCrop = explicitCrop
                } else {
                    // Default to full image resolution if no crop is provided
                    finalCrop = Bounds2i(pMin: Point2i(x: 0, y: 0), pMax: image.getResolution())
                }

                // 2. Prepare the pixel buffer
                var buffer = [Float]()

                func write(pixel: Pixel, at index: Int) {
                        buffer[index + 0] = Float(pixel.light.red)
                        buffer[index + 1] = Float(pixel.light.green)
                        buffer[index + 2] = Float(pixel.light.blue)
                        buffer[index + 3] = 1
                }

                let resolution = finalCrop.pMax - finalCrop.pMin
                buffer = [Float](repeating: 0, count: resolution.x * resolution.y * 4)
                for y in finalCrop.pMin.y..<finalCrop.pMax.y {
                        for x in finalCrop.pMin.x..<finalCrop.pMax.x {
                                let location = Point2i(x: x, y: y)
                                let pixel = image.getPixel(atLocation: location)
                                let px = x - finalCrop.pMin.x
                                let py = y - finalCrop.pMin.y
                                let index = py * resolution.x * 4 + px * 4
                                write(pixel: pixel, at: index)
                        }
                }
                
                // 3. OpenImageIO Tiling and Writing Logic
                
                let xres = Int32(resolution.x)
                let yres = Int32(resolution.y)
                let tileWidth = Int32(tileSize.0)
                let tileHeight = Int32(tileSize.1)
                let channels: Int32 = 4

                // Open the file
                guard let outputPointer = openImageForTiledWriting(
                    fileName, xres, yres, tileWidth, tileHeight, channels) else {
                    return
                }

                // Ensure the output pointer is closed and deleted when the function exits
                defer {
                    closeImageOutput(outputPointer)
                }

                // Stride Calculations
                let floatSize = Int64(MemoryLayout<Float>.size)
                let channelStride = floatSize
                let xStride = Int64(channels) * channelStride
                let yStride = Int64(xres) * xStride

                let nxtiles = (xres + tileWidth - 1) / tileWidth
                let nytiles = (yres + tileHeight - 1) / tileHeight
                
                // Tiling Loop
                buffer.withUnsafeBufferPointer { bufferPointer in
                    guard let pixels = bufferPointer.baseAddress else { return }
                    
                    for ty in 0..<nytiles {
                        for tx in 0..<nxtiles {
                            let success = writeSingleTile(
                                outputPointer,
                                pixels,
                                xres, channels,
                                Int32(tx), Int32(ty), tileWidth, tileHeight,
                                channelStride, xStride, yStride
                            )
                            
                            if !success {
                                return
                            }
                        }
                    }
                }
        }
}
