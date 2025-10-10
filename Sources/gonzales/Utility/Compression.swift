import Foundation

//import SWCompression

struct Compression {
        static func get(data: Data) throws -> Data {
                return data
                //return try GzipArchive.unarchive(archive: data)
        }
}
