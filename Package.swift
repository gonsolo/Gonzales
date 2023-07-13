// swift-tools-version:5.7

import PackageDescription

let package = Package(
        name: "gonzales",
        dependencies: [
                .package(
                        url: "https://github.com/tsolomko/SWCompression.git",
                        from: "4.8.0")
        ],
        targets: [
                .executableTarget(
                        name: "gonzales",
                        dependencies: [
                                "SWCompression",
                                "embree3",
                                "openImageIOBridge",
                                "cuda",
                                "cudaBridge",
                                "ptexBridge",
                        ]
                ),
                .target(
                        name: "openImageIOBridge",
                        dependencies: ["openimageio"]
                ),
                .target(
                        name: "ptexBridge",
                        dependencies: ["ptex"]
                ),
                .target(
                        name: "cudaBridge",
                        dependencies: ["cuda"]
                ),
                .systemLibrary(name: "embree3"),
                .systemLibrary(name: "openimageio", pkgConfig: "OpenImageIO"),
                .systemLibrary(name: "cuda", pkgConfig: "cuda-12.2"),
                .systemLibrary(name: "ptex", pkgConfig: "ptex"),
        ],
        cxxLanguageStandard: .cxx20
)
