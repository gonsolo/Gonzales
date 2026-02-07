import Foundation

struct Compression {
        static func get(data: Data) throws -> Data {
                return data
                // return try GzipArchive.unarchive(archive: data)
        }
}
