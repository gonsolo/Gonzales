///        A type that writes an image to a file.
protocol ImageWriter {
        func write(fileName: String, crop: Bounds2i, image: Image) throws
}
