import Foundation

enum MainError: Error {
        case format
        case missingOption
        case missingScene
        case unknownOption
}

func handle(_ error: Error) {
        switch error {
        case is ApiError:
                switch error {
                case ApiError.ply(let message):
                        print(message)
                case ApiError.unknownTextureFormat(let suffix):
                        print("Unknown texture format: \(suffix)")
                default:
                        print("ApiError: \(error)")
                }
        case is MainError:
                print("MainError: \(error)")
        case is RenderError:
                print("RenderError: \(error)")
        case is ImageError:
                print("ImageError \(error)")
        case is ParameterError:
                switch error {
                case ParameterError.missing(let parameter, let function):
                        print("Missing parameter: \(parameter) in function \(function)")
                case ParameterError.isNil:
                        print("isNil")
                default:
                        print("Parameter error!")
                }
        default:
                print("General error \(error)")
        }
}

func usage() {
        print("Usage: gonzales scene.pbrt")
}

func parseArguments() throws -> String {
        let arguments = Array(ProcessInfo.processInfo.arguments.dropFirst())
        if arguments.isEmpty {
                usage()
                exit(EXIT_SUCCESS)
        }
        var sceneName = ""
        var iterator = arguments.makeIterator()
        while let argument = iterator.next() {
                switch argument {
                case "--quick":
                        quick = true
                case "--verbose":
                        verbose = true
                case "--single":
                        guard let sx = iterator.next() else { throw MainError.missingOption }
                        guard let sy = iterator.next() else { throw MainError.missingOption }
                        guard let x = Int(sx) else { throw MainError.format }
                        guard let y = Int(sy) else { throw MainError.format }
                        singleRay = true
                        singleRayCoordinate = Point2I(x: x, y: y)
                case "--sync":
                        renderSynchronously = true
                case "--parse":
                        justParse = true
                case "--ptexmem":
                        guard let smem = iterator.next() else { throw MainError.missingOption }
                        guard let mem = Int(smem) else { throw MainError.format }
                        ptexMemory = mem
                case "--help":
                        usage()
                        exit(EXIT_SUCCESS)
                default:
                        sceneName = argument
                }
        }
        if sceneName.isEmpty {
                throw MainError.missingScene
        }
        return sceneName
}

func main() {
        let optix = Optix()
        optix.printCudaDevices()
        //do {
        //        let sceneName = try parseArguments()
        //        guard let sceneNameURL = URL(string: sceneName) else {
        //                throw RenderError.noSceneSpecified
        //        }
        //        let sceneNameLast = sceneNameURL.lastPathComponent
        //        var absoluteSceneName = ""
        //        if sceneName.starts(with: "/") {
        //                absoluteSceneName = sceneName
        //        } else {
        //                let fileManager = FileManager.default
        //                let currentDirectory = fileManager.currentDirectoryPath
        //                absoluteSceneName = currentDirectory + "/" + sceneName
        //        }
        //        let url = URL(fileURLWithPath: absoluteSceneName).deletingLastPathComponent()
        //        sceneDirectory = url.path
        //        api.start()
        //        try api.include(file: sceneNameLast, render: true)
        //} catch let error {
        //        handle(error)
        //}
}

main()
