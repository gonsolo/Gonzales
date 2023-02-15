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
                                "exr",
                                "embree3",
                                "oiio",
                                "openimageio",
                                "ptexBridge",
                        ],
                        linkerSettings: [
                                .unsafeFlags([])
                        ]
                ),
                .target(
                        name: "exr",
                        cxxSettings: [
                                .unsafeFlags([
                                        "-IExtern/openexr/src/lib/OpenEXR",
                                        "-IExtern/openexr/build/cmake",
                                        "-IExtern/openexr/build/_deps/imath-src/src/Imath",
                                        "-IExtern/openexr/build/_deps/imath-build/config",
                                        "-IExtern/openexr/src/lib/Iex",
                                ])
                        ],
                        linkerSettings: [
                                .unsafeFlags([
                                        "-LExtern/openexr/build/src/lib/OpenEXR",
                                        "-lOpenEXR-3_2",
                                        "-LExtern/openexr/build/_deps/imath-build/src/Imath",
                                        "-lImath-3_2",
                                ])
                        ]
                ),
                .target(
                        name: "oiio"
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
