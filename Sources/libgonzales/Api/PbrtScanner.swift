import Foundation  // Data, URL, pow, EOF, exit
import mojoKernel

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
                var cursor = Int32(bufferIndex)
                var result = Int32(0)
                guard mojo_scan_int(buffer, Int32(totalBytes), &cursor, &result) != 0 else {
                        return false
                }
                bufferIndex = Int(cursor)
                scanLocation = bufferIndex
                peekOne()
                intValue = Int(result)
                return true
        }

        func scanFloat(_ float: inout Float) throws -> Bool {
                var cursor = Int32(bufferIndex)
                var result = Float32(0)
                guard mojo_scan_float(buffer, Int32(totalBytes), &cursor, &result) != 0 else {
                        return false
                }
                bufferIndex = Int(cursor)
                scanLocation = bufferIndex
                peekOne()
                float = result
                return true
        }

        // Bulk scan: parse all floats from current position up to ']' / EOF.
        // Advances bufferIndex past the last parsed number (stops before ']').
        func scanFloats() -> [Float] {
                let count = Int(mojo_count_floats(buffer, Int32(totalBytes), Int32(bufferIndex)))
                guard count > 0 else { return [] }
                var out = [Float32](repeating: 0, count: count)
                var cursor = Int32(bufferIndex)
                let filled = out.withUnsafeMutableBufferPointer { ptr in
                        mojo_scan_floats(buffer, Int32(totalBytes), &cursor, ptr.baseAddress, Int32(count))
                }
                bufferIndex = Int(cursor)
                scanLocation = bufferIndex
                peekOne()
                return filled == Int32(count) ? out : Array(out[0..<Int(filled)])
        }

        // Bulk scan: parse all integers from current position up to ']' / EOF.
        func scanInts() -> [Int32] {
                let count = Int(mojo_count_ints(buffer, Int32(totalBytes), Int32(bufferIndex)))
                guard count > 0 else { return [] }
                var out = [Int32](repeating: 0, count: count)
                var cursor = Int32(bufferIndex)
                let filled = out.withUnsafeMutableBufferPointer { ptr in
                        mojo_scan_ints(buffer, Int32(totalBytes), &cursor, ptr.baseAddress, Int32(count))
                }
                bufferIndex = Int(cursor)
                scanLocation = bufferIndex
                peekOne()
                return filled == Int32(count) ? out : Array(out[0..<Int(filled)])
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

        var scanLocation = 0
        var isAtEnd = false
        var buffer: UnsafeMutablePointer<UInt8>
        let totalBytes: Int
        var bufferIndex: Int
        var currentByte: UInt8
}
