import cuda

func printCudaDevices() {
        var numDevices: Int32 = 0
        cudaGetDeviceCount(&numDevices)
        print("Cuda device count: \(numDevices)")
}
