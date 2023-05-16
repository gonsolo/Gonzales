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
                .systemLibrary(name: "embree3"),
                .systemLibrary(name: "openimageio", pkgConfig: "OpenImageIO"),
                .systemLibrary(name: "ptex", pkgConfig: "ptex"),
        ],
        cxxLanguageStandard: .cxx20
)
