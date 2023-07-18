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
                } catch (let error) {
                        fatalError("OptixError: \(error)")
                }
        }

        private func cudaCheck(_ cudaError: cudaError_t) throws {
                if cudaError != cudaSuccess {
                        throw OptixError.cudaCheck
                }
        }

        private func optixCheck(_ optixResult: OptixResult) throws {
                if optixResult != OPTIX_SUCCESS {
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

        func initializeOptix() throws {
                let optixResult = optixInit()
                try optixCheck(optixResult)
                print("Initializing Optix ok.")

                var cudaError: cudaError_t
                var stream: cudaStream_t?
                cudaError = cudaStreamCreate(&stream)
                try cudaCheck(cudaError)
                print("Cuda stream created.")
        }

        func dummy() {}

        static let shared = Optix()
}
