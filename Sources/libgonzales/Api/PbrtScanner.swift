import Foundation  // Data, URL, pow, EOF, exit

final class PbrtScanner {

        enum PbrtScannerError: Error {
                case decompress
                case noFile
                case unsupported
        }

        init(path: String) throws {

                // Load the entire (decompressed) file into one contiguous buffer. This
                // removes chunk-refill bookkeeping and lets the cursor be a plain index,
                // which the Mojo numeric scanner can operate on directly (see roadmap 11b).
                let data: Data
                if path.hasSuffix(".gz") {
                        guard let url = URL(string: "file://" + path) else {
                                throw PbrtScannerError.noFile
                        }
                        let raw = try Data(contentsOf: url)
                        data = try Compression.get(data: raw)
                } else {
                        guard let d = try? Data(contentsOf: URL(fileURLWithPath: path)) else {
                                throw PbrtScannerError.noFile
                        }
                        data = d
                }

                totalBytes = data.count
                buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: max(totalBytes, 1))
                if totalBytes > 0 {
                        data.copyBytes(to: buffer, count: totalBytes)
                }
                bufferIndex = 0
                currentByte = 0
        }

        deinit {
                buffer.deallocate()
        }

        func peekString(_ expected: String) -> String? {
                skipWhitespace()
                peekOne()
                let charString = ascii(currentByte)
                if charString != expected {
                        return nil
                } else {
                        return charString
                }
        }

        func scanString(_ expected: String) -> String? {
                skipWhitespace()
                peekOne()
                let charString = ascii(currentByte)
                if charString != expected {
                        return nil
                } else {
                        scanOne()
                        return charString
                }
        }

        func scanUpToString(_ input: String) -> String? {
                let scannedString = scanUpToCharactersList(from: [input])
                return scannedString

        }

        func scanInt(_ intValue: inout Int) -> Bool {
                skipWhitespace()
                peekOne()
                var isNegative = false
                if currentByte == minusChar {
                        isNegative = true
                        scanOne()
                }
                peekOne()
                if !isInteger(currentByte) {
                        return false
                }
                intValue = 0
                while isInteger(currentByte) {
                        scanOne()
                        intValue = 10 * intValue + (Int(currentByte) - 48)
                        peekOne()
                }
                if isNegative {
                        intValue = -intValue
                }
                return true
        }

        func scanFloat(_ float: inout Float) throws -> Bool {
                var intPart = 0
                // scanInt scans -0 as 0 so we have to remember whether we are negative
                skipWhitespace()
                peekOne()
                var isNegative = false
                if currentByte == minusChar {
                        isNegative = true
                }

                var doubleValue = 0.0
                var intSeen = false
                if scanInt(&intPart) {
                        doubleValue = Double(intPart)
                        intSeen = true
                }
                peekOne()
                if currentByte == dotChar {
                        scanOne()
                        var tenth = 0.1
                        peekOne()
                        while isInteger(currentByte) {
                                scanOne()
                                if doubleValue < 0 {
                                        doubleValue -= tenth * Double(currentByte - 48)
                                } else {
                                        doubleValue += tenth * Double(currentByte - 48)
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
                if currentByte == eChar {
                        scanOne()
                        if !scanInt(&exponent) {
                                exponent = 0
                        }
                        doubleValue *= pow(Double(10), Double(exponent))
                }

                float = Real(doubleValue)

                if isNegative && intPart == 0 {
                        float = -float
                }
                return true
        }

        func scanUpToCharactersList(from list: [String]) -> String? {
                var string = String()
                skipWhitespace()
                while true {
                        peekOne()
                        if currentByte == eofChar {
                                isAtEnd = true
                                return nil
                        }
                        let charString = ascii(currentByte)
                        if list.contains(charString) {
                                break
                        }
                        string.append(charString)
                        scanOne()
                }
                return string
        }

        private func ascii(_ byte: UInt8) -> String {
                return ascii(Int32(byte))
        }

        private func ascii(_ charCode: Int32) -> String {
                switch charCode {
                case EOF: return "EOF"
                case 0: return "EOF"
                case 9: return "\t"
                case 10: return "\n"
                case 13: return "\r"
                default:
                        guard let scalar = UnicodeScalar(Int(charCode)) else {
                                print(#function, "Unknown: ", charCode)
                                exit(0)
                        }
                        return String(scalar)
                }
        }

        private func peekOne() {
                if bufferIndex >= totalBytes {
                        currentByte = eofChar
                        return
                }
                currentByte = buffer[bufferIndex]
        }

        private func scanOne() {
                if bufferIndex >= totalBytes {
                        currentByte = eofChar
                        return
                }
                currentByte = buffer[bufferIndex]
                bufferIndex += 1
                scanLocation += 1
        }

        private func isInteger(_ byte: UInt8) -> Bool {
                if byte >= 48 && byte <= 57 {
                        return true
                } else {
                        return false
                }
        }

        private func isWhitespace(_ byte: UInt8) -> Bool {
                switch byte {
                case htabChar: return true
                case newlineChar: return true
                case spaceChar: return true
                default: return false
                }
        }

        private func skipWhitespace() {
                while true {
                        peekOne()
                        if !isWhitespace(currentByte) {
                                return
                        }
                        scanOne()
                }
        }

        let eofChar: UInt8 = 0
        let htabChar: UInt8 = 9
        let newlineChar: UInt8 = 10
        let spaceChar: UInt8 = 32
        let minusChar: UInt8 = 45
        let dotChar: UInt8 = 46
        let eChar: UInt8 = 101

        var scanLocation = 0
        var isAtEnd = false
        var buffer: UnsafeMutablePointer<UInt8>
        let totalBytes: Int
        var bufferIndex: Int
        var currentByte: UInt8
}
