import Foundation
import SWCompression

struct Compression {
        static func get(data: Data) throws -> Data {
                return try GzipArchive.unarchive(archive: data)
        }
}

