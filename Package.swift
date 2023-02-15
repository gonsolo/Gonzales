// swift-tools-version:5.7

import PackageDescription

let package = Package(
        name: "gonzales",
        platforms: [.macOS(.v10_15)],
        dependencies: [],
        targets: [
                .executableTarget(
                        name: "gonzales",
                        dependencies: [
                                "embree3",
                                "oiio",
                                "ptexBridge",
                        ],
                        linkerSettings: [
                                .unsafeFlags([])
                        ]
                ),
                .target(
                        name: "oiio",
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
