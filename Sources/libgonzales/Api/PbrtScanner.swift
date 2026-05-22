import Foundation  // Data, URL, Compression
import mojoKernel

final class PbrtScanner {

        enum PbrtScannerError: Error {
                case decompress
                case noFile
                case unsupported
        }

        init(path: String) throws {
                if path.hasSuffix(".gz") {
                        guard let url = URL(string: "file://" + path) else {
                                throw PbrtScannerError.noFile
                        }
                        let raw = try Data(contentsOf: url)
                        let data = try Compression.get(data: raw)
                        handle = data.withUnsafeBytes { ptr in
                                let p = ptr.bindMemory(to: UInt8.self).baseAddress!
                                return mojo_scanner_new_from_bytes(p, Int32(data.count))
                        }
                } else {
                        handle = path.withCString { cStr in
                                let p = UnsafeRawPointer(cStr).assumingMemoryBound(to: UInt8.self)
                                return mojo_scanner_new(p)
                        }
                }
                guard let h = handle else { throw PbrtScannerError.noFile }
                if mojo_scanner_is_at_end(h) != 0 { throw PbrtScannerError.noFile }
        }

        deinit {
                if let h = handle { mojo_scanner_free(h) }
        }

        var isAtEnd: Bool { mojo_scanner_is_at_end(handle) != 0 }
        var scanLocation: Int { Int(mojo_scanner_scan_location(handle)) }

        func peekString(_ expected: String) -> String? {
                guard expected.utf8.count == 1, let byte = expected.utf8.first else { return nil }
                return mojo_scanner_peek_char(handle, byte) != 0 ? expected : nil
        }

        func scanString(_ expected: String) -> String? {
                guard expected.utf8.count == 1, let byte = expected.utf8.first else { return nil }
                return mojo_scanner_scan_char(handle, byte) != 0 ? expected : nil
        }

        func scanUpToString(_ input: String) -> String? {
                return scanUpToCharactersList(from: [input])
        }

        func scanInt(_ intValue: inout Int) -> Bool {
                var result = Int32(0)
                guard mojo_scanner_scan_int(handle, &result) != 0 else { return false }
                intValue = Int(result)
                return true
        }

        func scanFloat(_ float: inout Float) throws -> Bool {
                var result = Float32(0)
                guard mojo_scanner_scan_float(handle, &result) != 0 else { return false }
                float = result
                return true
        }

        func scanFloats() -> [Float] {
                let count = Int(mojo_scanner_count_floats(handle))
                guard count > 0 else { return [] }
                var out = [Float32](repeating: 0, count: count)
                let filled = out.withUnsafeMutableBufferPointer { ptr in
                        mojo_scanner_scan_floats(handle, ptr.baseAddress, Int32(count))
                }
                return filled == Int32(count) ? out : Array(out[0..<Int(filled)])
        }

        func scanInts() -> [Int32] {
                let count = Int(mojo_scanner_count_ints(handle))
                guard count > 0 else { return [] }
                var out = [Int32](repeating: 0, count: count)
                let filled = out.withUnsafeMutableBufferPointer { ptr in
                        mojo_scanner_scan_ints(handle, ptr.baseAddress, Int32(count))
                }
                return filled == Int32(count) ? out : Array(out[0..<Int(filled)])
        }

        func parseQuotedString() -> String? {
                var buf = [UInt8](repeating: 0, count: 1024)
                let written = buf.withUnsafeMutableBufferPointer { ptr in
                        mojo_scanner_parse_quoted_string(handle, ptr.baseAddress, 1024)
                }
                guard written >= 0 else { return nil }
                return buf.withUnsafeBufferPointer { String(cString: $0.baseAddress!) }
        }

        func parseParamHeader() -> (String, String, Bool)? {
                var typeBuf = [UInt8](repeating: 0, count: 64)
                var nameBuf = [UInt8](repeating: 0, count: 128)
                var isArrayVal: Int32 = 0
                let found = typeBuf.withUnsafeMutableBufferPointer { typePtr in
                        nameBuf.withUnsafeMutableBufferPointer { namePtr in
                                mojo_scanner_parse_param_header(
                                        handle,
                                        typePtr.baseAddress, 64,
                                        namePtr.baseAddress, 128,
                                        &isArrayVal)
                        }
                }
                guard found != 0 else { return nil }
                let type = typeBuf.withUnsafeBufferPointer { String(cString: $0.baseAddress!) }
                let name = nameBuf.withUnsafeBufferPointer { String(cString: $0.baseAddress!) }
                return (type, name, isArrayVal != 0)
        }

        func scanUpToCharactersList(from list: [String]) -> String? {
                let delimBytes: [UInt8] = list.compactMap { $0.utf8.first }
                var buf = [UInt8](repeating: 0, count: 1024)
                let written: Int32 = delimBytes.withUnsafeBufferPointer { delimPtr in
                        buf.withUnsafeMutableBufferPointer { bufPtr in
                                mojo_scanner_scan_token(
                                        handle,
                                        delimPtr.baseAddress, Int32(delimBytes.count),
                                        bufPtr.baseAddress, 1024)
                        }
                }
                if written < 0 { return nil }
                if written == 0 { return isAtEnd ? nil : "" }
                return String(bytes: Array(buf[0..<Int(written)]), encoding: .utf8)
        }

        private let handle: OpaquePointer?
}
