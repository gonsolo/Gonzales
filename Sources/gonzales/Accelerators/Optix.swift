import Foundation
import cuda
import cudaBridge

//struct LaunchParams {
//        let dummy = 0
//}

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
        //var data: UnsafeMutableRawPointer? = nil
}

struct SimplePixel {
        let red: UInt8 = 0
        let green: UInt8 = 0
        let blue: UInt8 = 0
        let alpha: UInt8 = 0
}

struct PixelBlock2x2 {
        let pixels = (SimplePixel(), SimplePixel(), SimplePixel(), SimplePixel())
}

struct PixelBlock4x4 {
        let blocks = (PixelBlock2x2(), PixelBlock2x2(), PixelBlock2x2(), PixelBlock2x2())
}

struct PixelBlock8x8 {
        let blocks = (PixelBlock4x4(), PixelBlock4x4(), PixelBlock4x4(), PixelBlock4x4())
}

struct PixelBlock16x16 {
        let blocks = (PixelBlock8x8(), PixelBlock8x8(), PixelBlock8x8(), PixelBlock8x8())
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

        func download(_ t: inout T) throws {
                var t = t
                let error = cudaMemcpy(&t, pointer, 1, cudaMemcpyDeviceToHost)
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
                        try createMissPrograms()
                        try createHitgroupPrograms()
                        try createPipeline()
                        try buildShaderBindingTable()
                        //try launchParametersBuffer.alloc(size: MemoryLayout<LaunchParameters>.stride)
                        try colorBuffer.alloc(size: MemoryLayout<PixelBlock16x16>.stride)
                        //launchParameters.pointerToPixels = colorBuffer.pointer!
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

                var deviceContextOptions = OptixDeviceContextOptions()
                deviceContextOptions.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL

                var result = optixDeviceContextCreate(cudaContext, &deviceContextOptions, &optixContext)
                try optixCheck(result)
                result = optixDeviceContextSetLogCallback(
                        optixContext,
                        contextLogCallback,
                        nil,
                        4)
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
                //raygenRecord.data = nil
                try raygenRecordsBuffer.allocAndUpload(raygenRecord)
                shaderBindingTable.raygenRecord = raygenRecordsBuffer.devicePointer

                shaderBindingTable.exceptionRecord = 0
                shaderBindingTable.callablesRecordBase = 0
                shaderBindingTable.callablesRecordCount = 0
                shaderBindingTable.callablesRecordStrideInBytes = 16

                var missRecord = MissRecord()
                result = optixSbtRecordPackHeader(missProgramGroup, &missRecord)
                try optixCheck(result)
                try missRecordsBuffer.allocAndUpload(missRecord)
                shaderBindingTable.missRecordBase = missRecordsBuffer.devicePointer
                //print("miss stride: \(MemoryLayout<MissRecord>.stride)")
                shaderBindingTable.missRecordStrideInBytes = UInt32(MemoryLayout<MissRecord>.stride)
                shaderBindingTable.missRecordCount = 1

                var hitgroupRecord = HitgroupRecord()
                let result2 = optixSbtRecordPackHeader(hitgroupProgramGroup, &hitgroupRecord)
                try optixCheck(result2)
                //hitgroupRecord.objectID = 0
                try hitgroupRecordsBuffer.allocAndUpload(hitgroupRecord)
                shaderBindingTable.hitgroupRecordBase = hitgroupRecordsBuffer.devicePointer
                shaderBindingTable.hitgroupRecordStrideInBytes = UInt32(MemoryLayout<HitgroupRecord>.stride)
                shaderBindingTable.hitgroupRecordCount = 1
                printGreen("Optix shader binding table ok.")
        }

        func render() throws {
                printGreen("Optix render.")

                //let colorBuffer = CudaBuffer<UInt32>()
                //try colorBuffer.alloc(size: 16 * 16 * MemoryLayout<UInt32>.stride)
                //let opaquePointer = OpaquePointer(colorBuffer.pointer!)
                //let uint32Pointer = UnsafeMutablePointer<UInt32>(opaquePointer)
                //try launchParametersBuffer.alloc(size: MemoryLayout<LaunchParameters>.stride)

                //var blaPointer: UnsafeMutableRawPointer? = nil
                //let blaError = cudaMalloc(&blaPointer, 16 * 16 * 16)
                //try cudaCheck(blaError)
                //launchParameters.pointerToPixels = blaPointer
                let bla = buildLaunch()
                //launchParameters.frameId = 133
                //try launchParametersBuffer.upload(launchParameters)
                launchParameters.frameId += 1

                let width: UInt32 = 16
                let height: UInt32 = 16
                let depth: UInt32 = 1

                let result = optixLaunch(
                        pipeline,
                        stream,
                        //launchParametersBuffer.devicePointer,
                        //launchParametersBuffer.sizeInBytes,
                        bla,
                        16,
                        &shaderBindingTable,
                        width,
                        height,
                        depth)
                try optixCheck(result)
                printGreen("Optix render ok.")

                let color = getColor()
                print("color: ", color)

                cudaDeviceSynchronize()
                let error = cudaGetLastError()
                try cudaCheck(error)

                //var pixelBlock16x16 = PixelBlock16x16()
                //try colorBuffer.download(&pixelBlock16x16)
                //print(pixelBlock16x16.blocks.0.blocks.0.blocks.0.pixels.0.red)
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

        var raygenRecordsBuffer = CudaBuffer<RaygenRecord>()
        var missRecordsBuffer = CudaBuffer<MissRecord>()
        let hitgroupRecordsBuffer = CudaBuffer<HitgroupRecord>()
        //let launchParametersBuffer = CudaBuffer<LaunchParameters>()

        let colorBuffer = CudaBuffer<PixelBlock16x16>()

        var shaderBindingTable = OptixShaderBindingTable()
        var launchParameters = LaunchParameters()
}
