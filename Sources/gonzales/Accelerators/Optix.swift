import Foundation
import cuda
import cudaBridge

struct MissRecord {
        // force alignment of 16
        let dummy1 = 0
        let dummy2 = 0
}

struct HitgroupRecord {
        // force alignment of 16
        let dummy1 = 0
        let dummy2 = 0
}

enum OptixError: Error {
        case cudaCheck
        case noDevice
        case noFile
        case optixCheck
}

struct RaygenRecord {
        // force alignment of 16
        let dummy1 = 0
        let dummy2 = 0
}

struct SimplePixel {
        let red: UInt8 = 0
        let green: UInt8 = 0
        let blue: UInt8 = 0
        let alpha: UInt8 = 0
}

struct SimplePixel4: CustomStringConvertible {

        var description: String {
                return "SimplePixel4:\n  \(pixels.0)\n  \(pixels.1)\n  \(pixels.2)\n  \(pixels.3)"
        }

        subscript(x: Int, y: Int) -> SimplePixel {
                get {
                        let index = y * dimension + x
                        return self[index]
                }
        }

        subscript(index: Int) -> SimplePixel {
                get {
                        switch index {
                        case 0: return pixels.0
                        case 1: return pixels.1
                        case 2: return pixels.2
                        case 3: return pixels.3
                        default: fatalError("SimplePixel4: \(index)")
                        }
                }
        }

        var dimension: Int { 2 }
        var width: Int { dimension }
        var height: Int { dimension }
        var depth: Int { 1 }

        let pixels = (SimplePixel(), SimplePixel(), SimplePixel(), SimplePixel())
}

struct SimplePixel16: CustomStringConvertible {

        var description: String {
                return "SimplePixel16:\n  \(blocks.0)\n  \(blocks.1)\n  \(blocks.2)\n  \(blocks.3)"
        }

        subscript(x: Int, y: Int) -> SimplePixel {
                get {
                        let index = y * dimension + x
                        return self[index]
                }
        }

        subscript(index: Int) -> SimplePixel {
                get {
                        let quotient = index / 4
                        let remainder = index % 4

                        switch quotient {
                        case 0: return blocks.0[remainder]
                        case 1: return blocks.1[remainder]
                        case 2: return blocks.2[remainder]
                        case 3: return blocks.3[remainder]
                        default: fatalError("SimplePixel16: \(quotient)")
                        }
                }
        }

        var dimension: Int { 2 * blocks.0.dimension }
        var width: Int { dimension }
        var height: Int { dimension }
        var depth: Int { 1 }

        let blocks = (SimplePixel4(), SimplePixel4(), SimplePixel4(), SimplePixel4())
}

struct SimplePixel64: CustomStringConvertible {

        var description: String {
                return "SimplePixel64:\n  \(blocks.0)\n  \(blocks.1)\n  \(blocks.2)\n  \(blocks.3)"
        }

        subscript(x: Int, y: Int) -> SimplePixel {
                get {
                        let index = y * dimension + x
                        return self[index]
                }
        }

        subscript(index: Int) -> SimplePixel {
                get {
                        let quotient = index / 16
                        let remainder = index % 16

                        switch quotient {
                        case 0: return blocks.0[remainder]
                        case 1: return blocks.1[remainder]
                        case 2: return blocks.2[remainder]
                        case 3: return blocks.3[remainder]
                        default: fatalError("SimplePixel64: \(index) \(dimension) \(quotient) \(remainder)")
                        }
                }
        }

        var dimension: Int { 2 * blocks.0.dimension }
        var width: Int { dimension }
        var height: Int { dimension }
        var depth: Int { 1 }

        let blocks = (SimplePixel16(), SimplePixel16(), SimplePixel16(), SimplePixel16())
}

struct SimplePixel256: CustomStringConvertible {

        var description: String {
                return "SimplePixel256:\n  \(blocks.0)\n  \(blocks.1)\n  \(blocks.2)\n  \(blocks.3)"
        }

        subscript(x: Int, y: Int) -> SimplePixel {
                get {
                        let index = y * dimension + x
                        return self[index]
                }
        }

        subscript(index: Int) -> SimplePixel {
                get {
                        let quotient = index / 64
                        let remainder = index % 64

                        switch quotient {
                        case 0: return blocks.0[remainder]
                        case 1: return blocks.1[remainder]
                        case 2: return blocks.2[remainder]
                        case 3: return blocks.3[remainder]
                        default: fatalError("SimplePixel256: \(index) \(dimension) \(quotient) \(remainder)")
                        }
                }
        }

        var dimension: Int { 2 * blocks.0.dimension }
        var width: Int { dimension }
        var height: Int { dimension }
        var depth: Int { 1 }

        let blocks = (SimplePixel64(), SimplePixel64(), SimplePixel64(), SimplePixel64())
}

func cudaCheck(_ cudaError: cudaError_t) throws {
        if cudaError != cudaSuccess {
                throw OptixError.cudaCheck
        }
}

func cudaCheck(_ cudaResult: CUresult) throws {
        if cudaResult != CUDA_SUCCESS {
                throw OptixError.cudaCheck
        }
}

class CudaBuffer<T> {

        init() throws {
                let error = cudaMalloc(&pointer, sizeInBytes)
                try cudaCheck(error)
        }

        deinit {
                let error = cudaFree(&pointer)
                if error != cudaSuccess {
                        print("Error in \(self) deinit: cudaFree \(error)!")
                }
        }

        func download(_ t: inout T) throws {
                var t = t
                let error = cudaMemcpy(&t, pointer, sizeInBytes, cudaMemcpyDeviceToHost)
                try cudaCheck(error)

        }

        func upload(_ t: T) throws {
                var t = t
                let error = cudaMemcpy(pointer, &t, sizeInBytes, cudaMemcpyHostToDevice)
                try cudaCheck(error)
        }

        var sizeInBytes: Int {
                //return MemoryLayout<T>.stride
                return MemoryLayout<T>.size
        }

        var devicePointer: CUdeviceptr {
                return UInt64(bitPattern: Int64(Int(bitPattern: pointer)))
        }

        var pointer: UnsafeMutableRawPointer? = nil
}

class Optix {

        init() {
                do {
                        raygenRecordsBuffer = try CudaBuffer<RaygenRecord>()
                        missRecordsBuffer = try CudaBuffer<MissRecord>()
                        hitgroupRecordsBuffer = try CudaBuffer<HitgroupRecord>()
                        launchParametersBuffer = try CudaBuffer<LaunchParameters>()
                        colorBuffer = try CudaBuffer<PixelBlock>()
                        try initializeCuda()
                        try initializeOptix()
                        try createContext()
                        try createModule()
                        try createRaygenPrograms()
                        try createMissPrograms()
                        try createHitgroupPrograms()
                        try createPipeline()
                        try buildShaderBindingTable()
                } catch (let error) {
                        fatalError("OptixError: \(error)")
                }
        }

        private func optixCheck(_ optixResult: OptixResult) throws {
                if optixResult != OPTIX_SUCCESS {
                        print("OptixError: \(optixResult)")
                        throw OptixError.optixCheck
                }
        }

        private func cStringToString<T>(_ cString: T) -> String {
                return withUnsafePointer(to: cString) {
                        $0.withMemoryRebound(to: UInt8.self, capacity: MemoryLayout.size(ofValue: $0)) {
                                String(cString: $0)
                        }
                }
        }

        private func initializeCuda() throws {
                var numDevices: Int32 = 0
                var cudaError: cudaError_t
                cudaError = cudaGetDeviceCount(&numDevices)
                try cudaCheck(cudaError)
                guard numDevices == 1 else {
                        throw OptixError.noDevice
                }

                var cudaDevice: Int32 = 0
                cudaError = cudaGetDevice(&cudaDevice)
                try cudaCheck(cudaError)

                var cudaDeviceProperties: cudaDeviceProp = cudaDeviceProp()
                cudaError = cudaGetDeviceProperties_v2(&cudaDeviceProperties, cudaDevice)
                try cudaCheck(cudaError)

                let deviceName = cStringToString(cudaDeviceProperties.name)
                print(deviceName)
        }

        private func printGreen(_ message: String) {
                let escape = "\u{001B}"
                let bold = "1"
                let green = "32"
                let ansiEscapeGreen = escape + "[" + bold + ";" + green + "m"
                let ansiEscapeReset = escape + "[" + "0" + "m"
                print(ansiEscapeGreen + message + ansiEscapeReset)
        }

        private func initializeOptix() throws {
                let optixResult = optixInit()
                try optixCheck(optixResult)
                printGreen("Optix initialization ok.")
        }

        private func setLogger() throws {
                let logResult = optixDeviceContextSetLogCallback(
                        optixContext,
                        contextLogCallback,
                        nil,
                        4)
                try optixCheck(logResult)
        }

        private func createContext() throws {
                var cudaError: cudaError_t
                cudaError = cudaStreamCreate(&stream)
                try cudaCheck(cudaError)
                printGreen("Cuda stream ok.")

                var cudaResult: CUresult
                cudaResult = cuCtxGetCurrent(&cudaContext)
                try cudaCheck(cudaResult)
                printGreen("Cuda context ok.")

                var deviceContextOptions = OptixDeviceContextOptions()
                deviceContextOptions.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL

                let contextResult = optixDeviceContextCreate(
                        cudaContext,
                        &deviceContextOptions,
                        &optixContext)
                try optixCheck(contextResult)
                //try setLogger()
                printGreen("Optix context ok.")
        }

        private func getPipelineCompileOptions() -> OptixPipelineCompileOptions {
                var pipelineCompileOptions = OptixPipelineCompileOptions()
                pipelineCompileOptions.traversableGraphFlags =
                        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS.rawValue
                pipelineCompileOptions.usesMotionBlur = Int32(truncating: false)
                pipelineCompileOptions.numPayloadValues = 2
                pipelineCompileOptions.numAttributeValues = 2
                pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE.rawValue
                let launchParametersString = "launchParameters"
                launchParametersString.withCString {
                        pipelineCompileOptions.pipelineLaunchParamsVariableName = $0
                }
                return pipelineCompileOptions
        }

        private func createModule() throws {
                var moduleOptions = OptixModuleCompileOptions()
                moduleOptions.maxRegisterCount = 50
                moduleOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT
                moduleOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE

                var pipelineCompileOptions = getPipelineCompileOptions()

                let fileManager = FileManager.default
                let urlString = "file://" + fileManager.currentDirectoryPath + "/.build/kernels.optixir"
                guard let url = URL(string: urlString) else {
                        throw OptixError.noFile
                }
                let data = try Data(contentsOf: url)
                try data.withUnsafeBytes { input in
                        let inputSize = data.count
                        var logSize = 0
                        let optixResult = optixModuleCreate(
                                optixContext,
                                &moduleOptions,
                                &pipelineCompileOptions,
                                input.bindMemory(to: UInt8.self).baseAddress!,
                                inputSize,
                                nil,
                                &logSize,
                                &module)
                        try optixCheck(optixResult)
                }
                printGreen("Optix module ok.")
        }

        private func createRaygenPrograms() throws {
                var options = OptixProgramGroupOptions()
                var description = OptixProgramGroupDesc()
                description.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN
                description.raygen.module = module
                let raygenEntry = "__raygen__renderFrame"
                raygenEntry.withCString {
                        description.raygen.entryFunctionName = $0
                }
                let result = optixProgramGroupCreate(
                        optixContext,
                        &description,
                        1,
                        &options,
                        nil,
                        nil,
                        &raygenProgramGroup)
                try optixCheck(result)
                printGreen("Optix raygen ok.")
        }

        private func createMissPrograms() throws {

                var options = OptixProgramGroupOptions()
                var description = OptixProgramGroupDesc()
                description.kind = OPTIX_PROGRAM_GROUP_KIND_MISS
                description.miss.module = module
                let missRadiance = "__miss__radiance"
                missRadiance.withCString {
                        description.miss.entryFunctionName = $0
                }
                let result = optixProgramGroupCreate(
                        optixContext,
                        &description,
                        1,
                        &options,
                        nil,
                        nil,
                        &missProgramGroup)
                try optixCheck(result)
        }

        private func createHitgroupPrograms() throws {

                var options = OptixProgramGroupOptions()
                var description = OptixProgramGroupDesc()
                description.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP
                description.hitgroup.moduleCH = module
                let closesthitRadiance = "__closesthit__radiance"
                closesthitRadiance.withCString {
                        description.hitgroup.entryFunctionNameCH = $0
                }
                description.hitgroup.moduleAH = module
                let anyhitRadiance = "__anyhit__radiance"
                anyhitRadiance.withCString {
                        description.hitgroup.entryFunctionNameAH = $0
                }
                let result = optixProgramGroupCreate(
                        optixContext,
                        &description,
                        1,
                        &options,
                        nil,
                        nil,
                        &hitgroupProgramGroup)
                try optixCheck(result)
                printGreen("Optix hitgroup ok.")
        }

        private func createPipeline() throws {
                var pipelineCompileOptions = getPipelineCompileOptions()
                var pipelineLinkOptions = OptixPipelineLinkOptions()
                pipelineLinkOptions.maxTraceDepth = 2

                let result = optixPipelineCreate(
                        optixContext,
                        &pipelineCompileOptions,
                        &pipelineLinkOptions,
                        &raygenProgramGroup,
                        1,
                        nil,
                        nil,
                        &pipeline)
                try optixCheck(result)
                printGreen("Optix pipeline ok.")
        }

        private func buildShaderBindingTable() throws {
                var raygenRecord = RaygenRecord()
                var result = optixSbtRecordPackHeader(raygenProgramGroup, &raygenRecord)
                try optixCheck(result)
                try raygenRecordsBuffer.upload(raygenRecord)
                shaderBindingTable.raygenRecord = raygenRecordsBuffer.devicePointer

                shaderBindingTable.exceptionRecord = 0
                shaderBindingTable.callablesRecordBase = 0
                shaderBindingTable.callablesRecordCount = 0
                shaderBindingTable.callablesRecordStrideInBytes = 16

                var missRecord = MissRecord()
                result = optixSbtRecordPackHeader(missProgramGroup, &missRecord)
                try optixCheck(result)
                try missRecordsBuffer.upload(missRecord)
                shaderBindingTable.missRecordBase = missRecordsBuffer.devicePointer
                shaderBindingTable.missRecordStrideInBytes = UInt32(MemoryLayout<MissRecord>.stride)
                shaderBindingTable.missRecordCount = 1

                var hitgroupRecord = HitgroupRecord()
                let result2 = optixSbtRecordPackHeader(hitgroupProgramGroup, &hitgroupRecord)
                try optixCheck(result2)
                try hitgroupRecordsBuffer.upload(hitgroupRecord)
                shaderBindingTable.hitgroupRecordBase = hitgroupRecordsBuffer.devicePointer
                shaderBindingTable.hitgroupRecordStrideInBytes = UInt32(MemoryLayout<HitgroupRecord>.stride)
                shaderBindingTable.hitgroupRecordCount = 1
                printGreen("Optix shader binding table ok.")
        }

        func render() throws {
                printGreen("Optix render.")
                try buildLaunch()
                launchParameters.frameId += 1
                let result = optixLaunch(
                        pipeline,
                        stream,
                        launchParametersBuffer.devicePointer,
                        launchParametersBuffer.sizeInBytes,
                        &shaderBindingTable,
                        UInt32(pixelBlock.width),
                        UInt32(pixelBlock.height),
                        UInt32(pixelBlock.depth))
                try optixCheck(result)
                printGreen("Optix render ok.")
                cudaDeviceSynchronize()
                let error = cudaGetLastError()
                try cudaCheck(error)
                try printColors()
        }

        func printColors() throws {
                let error = cudaMemcpy(
                        &pixelBlock,
                        colorBuffer.pointer,
                        colorBuffer.sizeInBytes,
                        cudaMemcpyDeviceToHost)
                try cudaCheck(error)

                let resolution = Point2I(x: pixelBlock.width, y: pixelBlock.height)
                var image = Image(resolution: resolution)
                for y in 0..<pixelBlock.height {
                        for x in 0..<pixelBlock.width {
                                let color = RgbSpectrum(
                                        r: Float(pixelBlock[x, y].red) / 255,
                                        g: Float(pixelBlock[x, y].green) / 255,
                                        b: Float(pixelBlock[x, y].blue) / 255)
                                let pixel = Point2I(x: x, y: y)
                                image.addPixel(
                                        withColor: color,
                                        withWeight: 1,
                                        atLocation: pixel)
                        }
                }
                let imageWriter = OpenImageIOWriter()
                try imageWriter.write(fileName: "optix.exr", image: image)
        }

        func buildLaunch() throws {
                var launchParameters = LaunchParameters()
                launchParameters.width = Int32(pixelBlock.width)
                launchParameters.height = Int32(pixelBlock.height)
                launchParameters.pointerToPixels = colorBuffer.pointer
                let uploadError = cudaMemcpy(
                        launchParametersBuffer.pointer,
                        &launchParameters,
                        MemoryLayout<LaunchParameters>.stride,
                        cudaMemcpyHostToDevice)
                try cudaCheck(uploadError)
        }

        static let shared = Optix()

        var stream: cudaStream_t?
        var cudaContext: CUcontext?
        var optixContext: OptixDeviceContext?
        var pipeline: OptixPipeline?
        var module: OptixModule?

        var raygenProgramGroup: OptixProgramGroup?
        var missProgramGroup: OptixProgramGroup?
        var hitgroupProgramGroup: OptixProgramGroup?

        var raygenRecordsBuffer: CudaBuffer<RaygenRecord>
        var missRecordsBuffer: CudaBuffer<MissRecord>
        var hitgroupRecordsBuffer: CudaBuffer<HitgroupRecord>
        var launchParametersBuffer: CudaBuffer<LaunchParameters>

        var shaderBindingTable = OptixShaderBindingTable()
        var launchParameters = LaunchParameters()

        typealias PixelBlock = SimplePixel256
        var pixelBlock = PixelBlock()
        var colorBuffer: CudaBuffer<PixelBlock>
}
