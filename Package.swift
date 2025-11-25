// swift-tools-version:6.2

import PackageDescription

let package = Package(
        name: "gonzales",
        platforms: [
                .macOS(.v26)
        ],
        targets: [
                .executableTarget(
                        name: "gonzales",
                        dependencies: [
                                "openImageIOBridge",
                                "ptexBridge",
                        ],
                        swiftSettings: [
                                // 1. Swift-specific: Disables all runtime safety checks (bounds, overflow, preconditions)
                                .unsafeFlags(["-Ounchecked"]),

                                // 2. Clang Backend: High standard optimization
                                .unsafeFlags(["-Xcc", "-O3"]),

                                // 3. Clang Backend: Aggressive floating-point math optimizations (faster, less precise)
                                .unsafeFlags(["-Xcc", "-ffast-math"]),
                        ],
                ),
                .executableTarget(
                        name: "testCoated",
                ),
                .target(
                        name: "openImageIOBridge",
                        dependencies: ["openimageio"],
                        cxxSettings: [
                                .unsafeFlags(["-I/usr/local/include/"])
                        ],
                        swiftSettings: [.interoperabilityMode(.Cxx)]
                ),
                .target(
                        name: "ptexBridge",
                        dependencies: ["ptex"]
                ),
                .systemLibrary(name: "openimageio", pkgConfig: "OpenImageIO"),
                .systemLibrary(name: "ptex", pkgConfig: "ptex"),
        ],
        swiftLanguageModes: [.v6],
        cxxLanguageStandard: .cxx20
)
