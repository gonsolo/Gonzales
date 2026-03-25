import Foundation
import Testing

@testable import libgonzales

@Suite struct RenderRegressionTests {

        /// Renders a low-res cornell-box scene and compares against a reference image.
        @Test(.timeLimit(.minutes(1)))
        func cornellBoxRegression() async throws {
                let fileManager = FileManager.default
                let projectRoot = fileManager.currentDirectoryPath
                let scenePath = projectRoot + "/Tests/reference"
                let referencePath = projectRoot + "/Tests/reference/regression-test.exr"
                let outputPath = projectRoot + "/regression-test.exr"

                // Clean up any previous render
                try? fileManager.removeItem(atPath: outputPath)

                // Render the scene
                var renderOptions = RenderOptions()
                renderOptions.sceneDirectory = scenePath
                let sceneDescription = SceneDescription(renderOptions: renderOptions)
                sceneDescription.start()

                let clock = ContinuousClock()
                let elapsed = try await clock.measure {
                        try await sceneDescription.include(file: "regression-test.pbrt", render: true)
                }

                // Verify output was created
                #expect(fileManager.fileExists(atPath: outputPath), "Render did not produce output image")

                // Verify reference exists
                #expect(
                        fileManager.fileExists(atPath: referencePath),
                        "Reference image missing at \(referencePath)")

                // Compare images using magick
                let process = Process()
                process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
                process.arguments = [
                        "magick", "compare", "-metric", "RMSE", outputPath, referencePath, "null:",
                ]
                let pipe = Pipe()
                process.standardError = pipe  // magick writes metrics to stderr
                process.standardOutput = FileHandle.nullDevice
                try process.run()
                process.waitUntilExit()

                let data = pipe.fileHandleForReading.readDataToEndOfFile()
                let output = String(data: data, encoding: .utf8) ?? ""

                // Parse RMSE value from output like "123.45 (0.00123)"
                // The normalized value is in parentheses
                if let openParen = output.firstIndex(of: "("),
                        let closeParen = output.firstIndex(of: ")") {
                        let start = output.index(after: openParen)
                        let rmseString = String(output[start..<closeParen])
                        if let rmse = Double(rmseString) {
                                #expect(rmse < 0.1, "RMSE \(rmse) exceeds threshold 0.1")
                        }
                }

                // Check render time (128x128 @ 1spp should be very fast)
                let seconds = elapsed.components.seconds
                #expect(seconds < 30, "Render took \(seconds)s, exceeds 30s baseline")

                // Clean up rendered image
                try? fileManager.removeItem(atPath: outputPath)
        }
}
