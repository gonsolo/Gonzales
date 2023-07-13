import cuda

enum OptixError: Error {
        case noDevice
}

class Optix {

        init() {
                do {
                        try initializeCuda()
                        initializeOptix()
                } catch(let error) {
                        fatalError("OptixError: \(error)")
                }
        }

        private func cudaCheck(_ cudaError: cudaError_t) {
                if cudaError != cudaSuccess {
                        print("Cuda error: \(cudaError)")
                }
        }

        private func optixCheck(_ optixResult: OptixResult) {
                if optixResult != OPTIX_SUCCESS {
                        print("Optix error: \(optixResult)")
                }
        }

        func initializeCuda() throws {
                var numDevices: Int32 = 0
                var cudaError: cudaError_t
                cudaError = cudaGetDeviceCount(&numDevices)
                cudaCheck(cudaError)
                guard numDevices == 1 else {
                        throw OptixError.noDevice
                }

                var cudaDevice: Int32 = 0
                cudaError = cudaGetDevice(&cudaDevice)
                cudaCheck(cudaError)
                //print("Cuda device used: \(cudaDevice)")

                var cudaDeviceProperties: cudaDeviceProp = cudaDeviceProp()
                cudaError = cudaGetDeviceProperties_v2(&cudaDeviceProperties, cudaDevice)
                cudaCheck(cudaError)

                let deviceName = withUnsafePointer(to: cudaDeviceProperties.name) {
                        $0.withMemoryRebound(to: UInt8.self, capacity: MemoryLayout.size(ofValue: $0)) {
                                String(cString: $0)
                        }
                }
                print(deviceName)
        }

        func initializeOptix() {
                let optixResult = optixInit()
                optixCheck(optixResult)
                print("Initializing Optix ok.")

                var cudaError: cudaError_t
                var stream: cudaStream_t?
                cudaError = cudaStreamCreate(&stream)
                cudaCheck(cudaError)
        }

        func dummy() {}

        static let shared = Optix()
}
