import Foundation
import libgonzales
import mojoKernel

enum MainError: Error {
        case format
        case missingOption
        case missingScene
        case unknownOption
}

func handle(_ error: any Error) {
        switch error {
        case is SceneDescriptionError:
                switch error {
                case SceneDescriptionError.ply(let message):
                        print(message)
                case SceneDescriptionError.unknownTextureFormat(let suffix):
                        print("Unknown texture format: \(suffix)")
                default:
                        print("SceneDescriptionError: \(error)")
                }
        case is MainError:
                print("MainError: \(error)")
        case is RenderError:
                print("RenderError: \(error)")
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
        print("Usage: gonzales [options] scene.pbrt")
        print("Options:")
        print("  --quick             Quick render mode")
        print("  --interactive       Interactive viewer mode (1 spp)")
        print("  --gpu               Use GPU for BVH traversal")
        print("  --single X Y        Trace a single ray at pixel (X, Y)")
        print("  --parse             Just parse the scene without rendering")
        print("  --ptexmem MEM       Set Ptex memory limit (in MB/GB depending on implementation)")
        print("  --help              Show this help message")
}

@MainActor
func parseArguments() throws -> (String, RenderOptions) {
        let arguments = Array(ProcessInfo.processInfo.arguments.dropFirst())
        if arguments.isEmpty {
                usage()
                exit(EXIT_SUCCESS)
        }
        var sceneName = ""
        var renderOptions = RenderOptions()
        var iterator = arguments.makeIterator()
        while let argument = iterator.next() {
                switch argument {
                case "--quick":
                        renderOptions.quick = true
                case "--interactive":
                        renderOptions.interactive = true
                case "--gpu":
                        renderOptions.gpu = true
                case "--single":
                        guard let argumentX = iterator.next() else { throw MainError.missingOption }
                        guard let argumentY = iterator.next() else { throw MainError.missingOption }
                        guard let singleX = Int(argumentX) else { throw MainError.format }
                        guard let singleY = Int(argumentY) else { throw MainError.format }
                        renderOptions.singleRay = true
                        renderOptions.singleRayCoordinate = Point2i(x: singleX, y: singleY)
                case "--parse":
                        renderOptions.justParse = true
                case "--ptexmem":
                        guard let smem = iterator.next() else { throw MainError.missingOption }
                        guard let mem = Int(smem) else { throw MainError.format }
                        renderOptions.ptexMemory = mem
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
        return (sceneName, renderOptions)
}

@MainActor
func main() async {
        do {
                let (sceneName, parsedOptions) = try parseArguments()
                var renderOptions = parsedOptions
                guard let sceneNameURL = URL(string: sceneName) else {
                        throw RenderError.noSceneSpecified
                }
                let sceneNameLast = sceneNameURL.lastPathComponent
                var absoluteSceneName = ""
                if sceneName.starts(with: "/") {
                        absoluteSceneName = sceneName
                } else {
                        let fileManager = FileManager.default
                        let currentDirectory = fileManager.currentDirectoryPath
                        absoluteSceneName = currentDirectory + "/" + sceneName
                }
                let url = URL(fileURLWithPath: absoluteSceneName).deletingLastPathComponent()
                renderOptions.sceneDirectory = url.path
                let sceneDescription = SceneDescription(renderOptions: renderOptions)
                sceneDescription.start()
                try await sceneDescription.include(file: sceneNameLast, render: true)
        } catch let error {
                handle(error)
        }

        // Let the TileRenderer output settle before the overall timing
        try? await Task.sleep(nanoseconds: 100_000_000)
        let totalElapsed = DateInterval(start: globalStartTime, end: Date()).duration
        print("Gonzales Total Execution Time: \(totalElapsed.humanReadable)")
}

let globalStartTime = Date()
await main()
