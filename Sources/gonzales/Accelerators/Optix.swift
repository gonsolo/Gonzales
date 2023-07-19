import cuda

enum OptixError: Error {
        case cudaCheck
        case noDevice
        case optixCheck
}

class Optix {

        init() {
                do {
                        try initializeCuda()
                        try initializeOptix()
                        try createContext()
                        try createModule()
                } catch (let error) {
                        fatalError("OptixError: \(error)")
                }
        }

        private func cudaCheck(_ cudaError: cudaError_t) throws {
                if cudaError != cudaSuccess {
                        throw OptixError.cudaCheck
                }
        }

        private func cudaCheck(_ cudaResult: CUresult) throws {
                if cudaResult != CUDA_SUCCESS {
                        throw OptixError.cudaCheck
                }
        }

        private func optixCheck(_ optixResult: OptixResult) throws {
                if optixResult != OPTIX_SUCCESS {
                        print("OptixError, result: \(optixResult)")
                        throw OptixError.optixCheck
                }
        }

        func initializeCuda() throws {
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
                //print("Cuda device used: \(cudaDevice)")

                var cudaDeviceProperties: cudaDeviceProp = cudaDeviceProp()
                cudaError = cudaGetDeviceProperties_v2(&cudaDeviceProperties, cudaDevice)
                try cudaCheck(cudaError)

                let deviceName = withUnsafePointer(to: cudaDeviceProperties.name) {
                        $0.withMemoryRebound(to: UInt8.self, capacity: MemoryLayout.size(ofValue: $0)) {
                                String(cString: $0)
                        }
                }
                print(deviceName)
        }

        func printGreen(_ message: String) {
                let escape = "\u{001B}"
                let bold = "1"
                let green = "32"
                let ansiEscapeGreen = escape + "[" + bold + ";" + green + "m"
                let ansiEscapeReset = escape + "[" + "0" + "m"
                print(ansiEscapeGreen + message + ansiEscapeReset)
        }

        func initializeOptix() throws {
                let optixResult = optixInit()
                try optixCheck(optixResult)
                printGreen("Initializing Optix ok.")
        }

        func createContext() throws {
                var cudaError: cudaError_t
                cudaError = cudaStreamCreate(&stream)
                try cudaCheck(cudaError)
                printGreen("Cuda stream created.")

                var cudaResult: CUresult
                cudaResult = cuCtxGetCurrent(&cudaContext)
                try cudaCheck(cudaResult)
                printGreen("Cuda context ok.")

                let optixResult = optixDeviceContextCreate(cudaContext, nil, &optixContext)
                try optixCheck(optixResult)
                printGreen("Optix context ok.")
        }

        func createModule() throws {
                var moduleOptions = OptixModuleCompileOptions()
                var pipelineOptions = OptixPipelineCompileOptions()
                let input = ""
                let inputSize = input.count
                var logSize = 0
                var module: OptixModule? = nil
                let optixResult = optixModuleCreate(
                        optixContext,
                        &moduleOptions,
                        &pipelineOptions,
                        input,
                        inputSize,
                        nil,
                        &logSize,
                        &module)
                try optixCheck(optixResult)

        }

        func dummy() {}

        static let shared = Optix()

        var stream: cudaStream_t?
        var cudaContext: CUcontext?
        var optixContext: OptixDeviceContext?
}
