import Foundation
import cuda

struct LaunchParams {}

enum OptixError: Error {
        case cudaCheck
        case noDevice
        case noFile
        case optixCheck
}

struct RaygenRecord {

        var data: UnsafeMutableRawPointer? = nil
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

        func alloc(size: Int) throws {
                sizeInBytes = size
                let error = cudaMalloc(&pointer, sizeInBytes)
                try cudaCheck(error)
        }

        func upload(_ t: T) throws {
                var t = t
                let error = cudaMemcpy(pointer, &t, 1, cudaMemcpyHostToDevice)
                try cudaCheck(error)
        }

        func allocAndUpload(_ t: T) throws {
                try alloc(size: MemoryLayout<T>.stride)
                try upload(t)
        }

        var devicePointer: CUdeviceptr {
                guard let nonNilAddress = pointer else {
                        fatalError("Nil!")
                }
                return CUdeviceptr(UInt(bitPattern: nonNilAddress))
        }

        var sizeInBytes = 0
        var pointer: UnsafeMutableRawPointer? = nil
}

class Optix {

        init() {
                do {
                        try initializeCuda()
                        try initializeOptix()
                        try createContext()
                        try createModule()
                        try createRaygenPrograms()
                        try createPipeline()
                        try buildShaderBindingTable()
                        try launchParamsBuffer.alloc(size: MemoryLayout<LaunchParams>.stride)
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

        private func createContext() throws {
                var cudaError: cudaError_t
                cudaError = cudaStreamCreate(&stream)
                try cudaCheck(cudaError)
                printGreen("Cuda stream ok.")

                var cudaResult: CUresult
                cudaResult = cuCtxGetCurrent(&cudaContext)
                try cudaCheck(cudaResult)
                printGreen("Cuda context ok.")

                let optixResult = optixDeviceContextCreate(cudaContext, nil, &optixContext)
                try optixCheck(optixResult)
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
                let optixLaunchParams = "optixLaunchParams"
                optixLaunchParams.withCString {
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
                let result = optixSbtRecordPackHeader(raygenProgramGroup, &raygenRecord)
                try optixCheck(result)
                raygenRecord.data = nil
                try raygenRecordsBuffer.allocAndUpload(raygenRecord)
                shaderBindingTable.raygenRecord = raygenRecordsBuffer.devicePointer
                printGreen("Optix shader binding table ok.")
        }

        func render() throws {
                printGreen("Optix render.")
                try launchParamsBuffer.upload(launchParams)

                let width: UInt32 = 10
                let height: UInt32 = 10
                let depth: UInt32 = 1

                let result = optixLaunch(
                        pipeline,
                        stream,
                        launchParamsBuffer.devicePointer,
                        launchParamsBuffer.sizeInBytes,
                        &shaderBindingTable,
                        width,
                        height,
                        depth)
                try optixCheck(result)
                printGreen("Optix render ok.")
        }

        static let shared = Optix()

        var stream: cudaStream_t?
        var cudaContext: CUcontext?
        var optixContext: OptixDeviceContext?
        var pipeline: OptixPipeline?
        var module: OptixModule?
        var raygenProgramGroup: OptixProgramGroup?
        var raygenRecordsBuffer = CudaBuffer<RaygenRecord>()
        var shaderBindingTable = OptixShaderBindingTable()
        let launchParams = LaunchParams()
        let launchParamsBuffer = CudaBuffer<LaunchParams>()
}
