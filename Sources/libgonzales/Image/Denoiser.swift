import openimagedenoise

/// AI-powered denoiser using Intel Open Image Denoise.
/// Takes noisy beauty image + albedo/normal auxiliary buffers and produces
/// a clean result using OIDN's "RT" filter (trained for path-traced images).
struct Denoiser {

        /// Denoise the beauty image in-place using albedo and normal as auxiliary inputs.
        /// All images must have the same resolution.
        static func denoise(
                beauty: inout Image,
                albedo: Image,
                normal: Image
        ) {
                let resolution = beauty.getResolution()
                let width = resolution.x
                let height = resolution.y
                let pixelCount = width * height

                // Convert Image → flat [Float] RGB arrays (3 floats per pixel)
                var colorBuffer = [Float](repeating: 0, count: pixelCount * 3)
                var albedoBuffer = [Float](repeating: 0, count: pixelCount * 3)
                var normalBuffer = [Float](repeating: 0, count: pixelCount * 3)

                for y in 0..<height {
                        for x in 0..<width {
                                let loc = Point2i(x: x, y: y)
                                let idx = (y * width + x) * 3

                                let bp = beauty.getPixel(atLocation: loc)
                                colorBuffer[idx + 0] = Float(bp.light.red)
                                colorBuffer[idx + 1] = Float(bp.light.green)
                                colorBuffer[idx + 2] = Float(bp.light.blue)

                                let ap = albedo.getPixel(atLocation: loc)
                                albedoBuffer[idx + 0] = Float(ap.light.red)
                                albedoBuffer[idx + 1] = Float(ap.light.green)
                                albedoBuffer[idx + 2] = Float(ap.light.blue)

                                let np = normal.getPixel(atLocation: loc)
                                normalBuffer[idx + 0] = Float(np.light.red)
                                normalBuffer[idx + 1] = Float(np.light.green)
                                normalBuffer[idx + 2] = Float(np.light.blue)
                        }
                }

                var outputBuffer = [Float](repeating: 0, count: pixelCount * 3)

                // Create OIDN device (auto-selects best: CUDA > HIP > CPU)
                let device = oidnNewDevice(OIDN_DEVICE_TYPE_DEFAULT)
                oidnCommitDevice(device)

                // Check device creation
                var errorMessage: UnsafePointer<CChar>?
                let deviceError = oidnGetDeviceError(device, &errorMessage)
                if deviceError != OIDN_ERROR_NONE {
                        if let msg = errorMessage {
                                print("OIDN device error: \(String(cString: msg))")
                        }
                        oidnReleaseDevice(device)
                        return
                }

                // Print device info
                let deviceType = oidnGetDeviceInt(device, "type")
                let deviceTypeStr: String
                switch deviceType {
                case 1: deviceTypeStr = "CPU"
                case 2: deviceTypeStr = "SYCL"
                case 3: deviceTypeStr = "CUDA"
                case 4: deviceTypeStr = "HIP"
                case 5: deviceTypeStr = "Metal"
                default: deviceTypeStr = "Unknown(\(deviceType))"
                }
                print("OIDN: Using \(deviceTypeStr) device, denoising \(width)x\(height) image")

                // Create RT filter
                let filter = oidnNewFilter(device, "RT")

                // Set images — use shared memory (zero-copy)
                        let denoiseTimer = Timer("Denoising...", newline: false)
                        colorBuffer.withUnsafeMutableBufferPointer { colorPtr in
                                albedoBuffer.withUnsafeMutableBufferPointer { albedoPtr in
                                        normalBuffer.withUnsafeMutableBufferPointer { normalPtr in
                                                outputBuffer.withUnsafeMutableBufferPointer { outputPtr in

                                                        oidnSetSharedFilterImage(
                                                                filter, "color",
                                                                colorPtr.baseAddress,
                                                                OIDN_FORMAT_FLOAT3,
                                                                size_t(width), size_t(height),
                                                                0, 0, 0)

                                                        oidnSetSharedFilterImage(
                                                                filter, "albedo",
                                                                albedoPtr.baseAddress,
                                                                OIDN_FORMAT_FLOAT3,
                                                                size_t(width), size_t(height),
                                                                0, 0, 0)

                                                        oidnSetSharedFilterImage(
                                                                filter, "normal",
                                                                normalPtr.baseAddress,
                                                                OIDN_FORMAT_FLOAT3,
                                                                size_t(width), size_t(height),
                                                                0, 0, 0)

                                                        oidnSetSharedFilterImage(
                                                                filter, "output",
                                                                outputPtr.baseAddress,
                                                                OIDN_FORMAT_FLOAT3,
                                                                size_t(width), size_t(height),
                                                                0, 0, 0)

                                                        oidnSetFilterBool(filter, "hdr", true)
                                                        oidnSetFilterInt(filter, "quality", 6)  // OIDN_QUALITY_HIGH

                                                        oidnCommitFilter(filter)
                                                        oidnExecuteFilter(filter)
                                                }
                                        }
                                }
                        }

                        // Check for errors
                        let filterError = oidnGetDeviceError(device, &errorMessage)
                        if filterError != OIDN_ERROR_NONE {
                                if let msg = errorMessage {
                                        print("\nOIDN filter error: \(String(cString: msg))")
                                }
                        } else {
                                // Copy denoised result back into the beauty Image
                                for y in 0..<height {
                                        for x in 0..<width {
                                                let idx = (y * width + x) * 3
                                                let color = RgbSpectrum(
                                                        red: Real(outputBuffer[idx + 0]),
                                                        green: Real(outputBuffer[idx + 1]),
                                                        blue: Real(outputBuffer[idx + 2])
                                                )
                                                beauty.setPixel(color: color, atLocation: Point2i(x: x, y: y))
                                        }
                                }
                                let duration = denoiseTimer.duration
                                print(String(format: "\rOIDN: Denoising complete (%.3fs)", duration))
                        }

                oidnReleaseFilter(filter)
                oidnReleaseDevice(device)
        }
}
