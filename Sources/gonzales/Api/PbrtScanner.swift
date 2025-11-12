import Foundation  // InputStream, pow, EOF, exit

final class PbrtScanner {

        enum PbrtScannerError: Error {
                case decompress
                case noFile
                case unsupported
        }

        init(path: String) throws {

                if path.hasSuffix(".gz") {
                        let urlString = "file://" + path
                        guard let url = URL(string: urlString) else {
                                throw PbrtScannerError.noFile
                        }
                        let data = try Data(contentsOf: url)
                        let decompressedData = try Compression.get(data: data)
                        let inputStream = InputStream(data: decompressedData)
                        self.stream = inputStream
                } else {
                        guard let inputStream = InputStream(fileAtPath: path) else {
                                throw PbrtScannerError.noFile
                        }
                        self.stream = inputStream
                }
                stream.open()
                if stream.streamStatus == .error {
                        throw PbrtScannerError.noFile
                }
                var bytes: [UInt8] = Array(repeating: 0, count: bufferLength)
                buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: bufferLength)
                buffer.initialize(from: &bytes, count: bufferLength)
                bufferIndex = 0
                bytesRead = stream.read(buffer, maxLength: bufferLength)
                c = 0
        }

        deinit {
                buffer.deallocate()
        }

        func peekString(_ expected: String) -> String? {
                skipWhitespace()
                peekOne()
                let s = ascii(c)
                if s != expected {
                        return nil
                } else {
                        return s
                }
        }

        func scanString(_ expected: String) -> String? {
                skipWhitespace()
                peekOne()
                let s = ascii(c)
                if s != expected {
                        return nil
                } else {
                        scanOne()
                        return s
                }
        }

        func scanUpToString(_ input: String) -> String? {
                let s = scanUpToCharactersList(from: [input])
                return s

        }

        func scanInt(_ i: inout Int) -> Bool {
                skipWhitespace()
                peekOne()
                var isNegative = false
                if c == minus {
                        isNegative = true
                        scanOne()
                }
                peekOne()
                if !isInteger(c) {
                        return false
                }
                i = 0
                while isInteger(c) {
                        scanOne()
                        i = 10 * i + (Int(c) - 48)
                        peekOne()
                }
                if isNegative {
                        i = -i
                }
                return true
        }

        func scanFloat(_ float: inout Float) throws -> Bool {
                var i = 0
                // scanInt scans -0 as 0 so we have to remember whether we are negative
                skipWhitespace()
                peekOne()
                var isNegative = false
                if c == minus {
                        isNegative = true
                }

                var f = 0.0
                var intSeen = false
                if scanInt(&i) {
                        f = Double(i)
                        intSeen = true
                }
                peekOne()
                if c == dot {
                        scanOne()
                        var tenth = 0.1
                        peekOne()
                        while isInteger(c) {
                                scanOne()
                                if f < 0 {
                                        f -= tenth * Double(c - 48)
                                } else {
                                        f += tenth * Double(c - 48)
                                }
                                tenth *= 0.1
                                peekOne()
                        }
                } else {
                        // If neither a number not a dot is seen this is not a floating point number
                        if !intSeen {
                                return false
                        }
                }
                peekOne()
                var exponent = 0
                if c == e {
                        scanOne()
                        if !scanInt(&exponent) {
                                exponent = 0
                        }
                        f = f * pow(Double(10), Double(exponent))
                }

                float = FloatX(f)

                if isNegative && i == 0 {
                        float = -float
                }
                return true
        }

        func scanUpToCharactersList(from list: [String]) -> String? {
                var string = String()
                skipWhitespace()
                while true {
                        peekOne()
                        if c == eof {
                                isAtEnd = true
                                return nil
                        }
                        let s = ascii(c)
                        if match(character: s, in: list) {
                                break
                        }
                        string.append(s)
                        scanOne()
                }
                return string
        }

        private func match(character: String, in list: [String]) -> Bool {
                for l in list {
                        if character == l { return true }
                }
                return false
        }

        private func ascii(_ x: UInt8) -> String {
                return ascii(Int32(x))
        }

        private func ascii(_ x: Int32) -> String {
                switch x {
                case EOF: return "EOF"
                case 0: return "EOF"
                case 9: return "\t"
                case 10: return "\n"
                case 13: return "\r"
                default:
                        guard let scalar = UnicodeScalar(Int(x)) else {
                                print(#function, "Unknown: ", x)
                                exit(0)
                        }
                        return String(scalar)
                }
        }

        private func peekOne() {
                if bytesRead == 0 {
                        return
                }
                c = buffer[bufferIndex]
        }

        private func scanOne() {
                if bytesRead == 0 {
                        c = eof
                        return
                }
                c = buffer[bufferIndex]
                bufferIndex += 1
                scanLocation += 1
                if bufferIndex == bytesRead {
                        bufferIndex = 0
                        bytesRead = stream.read(buffer, maxLength: bufferLength)
                }
        }

        private func isInteger(_ c: UInt8) -> Bool {
                if c >= 48 && c <= 57 {
                        return true
                } else {
                        return false
                }
        }

        private func isWhitespace(_ c: UInt8) -> Bool {
                switch c {
                case htab: return true
                case newline: return true
                case space: return true
                default: return false
                }
        }

        private func skipWhitespace() {
                while true {
                        peekOne()
                        if !isWhitespace(c) {
                                return
                        }
                        scanOne()
                }
        }

        let eof: UInt8 = 0
        let htab: UInt8 = 9
        let newline: UInt8 = 10
        let space: UInt8 = 32
        let minus: UInt8 = 45
        let dot: UInt8 = 46
        let e: UInt8 = 101

        var scanLocation = 0
        var isAtEnd = false
        var bytesRead: Int
        var buffer: UnsafeMutablePointer<UInt8>
        let bufferLength = 64 * 1024
        var bufferIndex: Int
        var stream: InputStream
        var c: UInt8
}
