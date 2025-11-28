// swift-tools-version:6.2

import PackageDescription

let package = Package(
        name: "gonzales",
        platforms: [
                .macOS(.v26)
        ],
        targets: [
                .target(
                        name: "libgonzales",
                        dependencies: [
                                "openImageIOBridge",
                                "ptexBridge",
                        ],
                        swiftSettings: [
                                .unsafeFlags(["-Ounchecked"]),
                                .unsafeFlags(["-Xcc", "-O3"]),
                                .unsafeFlags(["-Xcc", "-ffast-math"]),
                        ]
                ),
                .testTarget(
                        name: "libgonzalesTests",
                        dependencies: [
                                "libgonzales"
                        ]
                ),
                .executableTarget(
                        name: "gonzales",
                        dependencies: [
                                "libgonzales",
                        ],
                        swiftSettings: [
                                .unsafeFlags(["-Ounchecked"]),
                                .unsafeFlags(["-Xcc", "-O3"]),
                                .unsafeFlags(["-Xcc", "-ffast-math"]),
                        ],
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
