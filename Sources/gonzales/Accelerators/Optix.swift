import Foundation
import cuda
import cudaBridge

enum OptixError: Error {
        case cudaCheck
        case noDevice
        case noFile
        case optixCheck
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

func optixError(_ message: String = "") -> Never {
        print("optixError: \(message)")
        exit(-1)
}

class CudaBuffer<T> {

        init(count: Int = 1) throws {
                self.count = count
                try allocate()
        }

        deinit {
                let error = cudaFree(pointer)
                if error != cudaSuccess {
                        print("Error in \(self) deinit: cudaFree \(error)!")
                }
        }

        private func allocate() throws {
                let error = cudaMalloc(&pointer, sizeInBytes)
                try cudaCheck(error)
        }

        func download(_ t: inout T) throws {
                try withUnsafeMutablePointer(to: &t) { t in
                        let error = cudaMemcpy(t, pointer, sizeInBytes, cudaMemcpyDeviceToHost)
                        try cudaCheck(error)
                }
        }

        func upload(_ t: T) throws {
                var t = t
                try withUnsafePointer(to: &t) { t in
                        let error = cudaMemcpy(pointer, t, sizeInBytes, cudaMemcpyHostToDevice)
                        try cudaCheck(error)
                }
        }

        var sizeInBytes: Int {
                //return count * MemoryLayout<T>.stride
                return count * MemoryLayout<T>.size
        }

        var devicePointer: CUdeviceptr {
                return UInt64(bitPattern: Int64(Int(bitPattern: pointer)))
        }

        var count: Int = 0
        var pointer: UnsafeMutableRawPointer? = nil
}

class Optix {

       var triangleInput = OptixBuildInput()

        private func add(triangle: Triangle) throws {

                let points = triangle.getWorldPoints()

                bounds = union(first: bounds, second: triangle.worldBound())

                gonzoAdd(
                        points.0.x, points.0.y, points.0.z,
                        points.1.x, points.1.y, points.1.z,
                        points.2.x, points.2.y, points.2.z)
        }

        var deviceVertices: CUdeviceptr = 0
        var triangleInputFlags: UInt32 = 0

        private func syncCheck() throws {
                cudaDeviceSynchronize()
                let lastError = cudaGetLastError()
                try cudaCheck(lastError)
        }

        var triangleCount = 0

        func add(primitives: [Boundable & Intersectable]) throws {
                //var triangleInput: OptixBuildInput! = nil
                for primitive in primitives {
                        switch primitive {
                        case let geometricPrimitive as GeometricPrimitive:
                                switch geometricPrimitive.shape {
                                case let triangle as Triangle:
                                                try add(triangle: triangle)
                                                materials[triangleCount] = geometricPrimitive.material
                                                triangleCount += 1
                                default:
                                        var message = "Unknown shape in geometric primitive: "
                                        message += "\(geometricPrimitive.shape)"
                                        warnOnce(message)
                                }
                        case let areaLight as AreaLight:
                                switch areaLight.shape {
                                case let triangle as Triangle:
                                                try add(triangle: triangle)
                                                areaLights[triangleCount] = areaLight
                                                triangleCount += 1
                                        //_ = triangle  // TODO
                                default:
                                        optixError("Unknown shape in AreaLight.")
                                }
                        default:
                                optixError("Unknown primitive \(primitive).")
                        }
                }
        //        traversableHandle = try buildAccel()
        //        printGreen("Optix: Added \(triangleCount) triangles.")
        }

        var traversableHandle: OptixTraversableHandle = 0

        private func optixCheck(_ optixResult: OptixResult, _ lineNumber: Int = #line) throws {
                if optixResult != OPTIX_SUCCESS {
                        print("OptixError: \(optixResult) from line \(lineNumber)")
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

                cudaFree(nil)

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
                cudaError = cudaSetDevice(cudaDevice)
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

        func optixSetup() {
                gonzoSetup()
        }

        func optixRender() {
                var px: Float32 = 0
                var py: Float32 = 0
                var pz: Float32 = 0
                var nx: Float32 = 0
                var ny: Float32 = 0
                var nz: Float32 = 0
                var tHit: Float = 1e20
                var intersected: Int32 = 0
                var primID: Int32 = -1
                gonzoRender(false, 0, 0, 0, 0, 0, 0, &tHit, &px, &py, &pz, &nx, &ny, &nz, &intersected, &primID)
        }

        func optixRender(ray: Ray, tHit: inout Float) -> (Point, Normal, Bool, Int) {
                var px: Float32 = 0
                var py: Float32 = 0
                var pz: Float32 = 0
                var nx: Float32 = 0
                var ny: Float32 = 0
                var nz: Float32 = 0
                var intersected: Int32 = 0
                var primID32: Int32 = -1
                gonzoRender(
                        true,
                        ray.origin.x, ray.origin.y, ray.origin.z,
                        ray.direction.x, ray.direction.y, ray.direction.z,
                        &tHit,
                        &px, &py, &pz,
                        &nx, &ny, &nz,
                        &intersected,
                        &primID32
                        )
                let intersectionPoint = Point(x: px, y: py, z: pz)
                let intersectionNormal = Normal(x: nx, y: ny, z: nz)
                let intersectionIntersected: Bool = intersected == 1 ? true : false
                let primID = Int(primID32)
                return (intersectionPoint, intersectionNormal, intersectionIntersected, primID)
        }

        func optixWrite() {
                gonzoWrite()
        }

        func render(ray: Ray, tHit: inout Float) throws -> (Point, Normal, Bool, Int) {
                return optixRender(ray: ray, tHit: &tHit);
        }

        func intersect(
                ray: Ray,
                tHit: inout FloatX,
                material: MaterialIndex,
                interaction: inout SurfaceInteraction
        ) throws {
                let (intersectionPoint, intersectionNormal, intersected, primID) = try render(ray: ray, tHit: &tHit)
                if intersected {
                        interaction.valid = true
                        interaction.position = intersectionPoint
                        interaction.normal = intersectionNormal
                        interaction.shadingNormal = intersectionNormal
                        interaction.wo = -ray.direction
                        // dpdu
                        // uv
                        // faceIndex
                        interaction.material = materials[primID] ?? -1
                        interaction.areaLight = areaLights[primID] ?? nil
                }

        }

        func objectBound() -> Bounds3f {
                return bounds
        }

        func worldBound() -> Bounds3f {
                return bounds
        }

        var launchParametersBuffer: CudaBuffer<LaunchParameters>! = nil

        var bounds = Bounds3f()

        var materials = [Int: MaterialIndex]()
        var areaLights = [Int: AreaLight]()
}

