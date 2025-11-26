import Foundation

struct AsciiWriter {

        func write(fileName: String, crop: Bounds2i, image: Image) throws {

                let resolution = crop.pMax - crop.pMin
                var text = String(resolution.x) + " " + String(resolution.y) + "\n"

                func write(pixel: Pixel, at _: Int) {
                        text.append(String(pixel.light.red))
                        text.append(" ")
                        text.append(String(pixel.light.green))
                        text.append(" ")
                        text.append(String(pixel.light.blue))
                        text.append(" ")
                        text.append("1.0")
                        text.append(" ")
                }

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

                let url = URL(fileURLWithPath: "/home/gonsolo/" + fileName)
                print(url)
                do {
                        try text.write(to: url, atomically: true, encoding: String.Encoding.utf8)
                } catch {
                        print("Error writing file")
                }
        }
}
