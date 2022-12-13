// swift-tools-version:5.7

import PackageDescription

let package = Package(
        name: "gonzales",
        platforms: [.macOS(.v10_15)],
        dependencies: [
                .package(
                        url: "https://github.com/kelvin13/png.git",
                        from: "4.0.2")
        ],
        targets: [
                .executableTarget(
                        name: "gonzales",
                        dependencies: [
                                .product(
                                        name: "PNG",
                                        package: "png"),
                                "exr",
                                "embree",
                                "ptex",
                        ],
                        linkerSettings: [
                                .unsafeFlags([
                                        "-LExtern/ptex/build/src/ptex",
                                        "-lPtex",
                                ])
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
                        name: "ptex",
                        dependencies: [],
                        cxxSettings: [
                                .unsafeFlags([
                                        "-IExtern/ptex/src/ptex"
                                ])
                        ]),
                .target(
                        name: "embree",
                        dependencies: [],
                        cxxSettings: [],
                        linkerSettings: [
                                .unsafeFlags([
                                        "-lembree3"
                                ])
                        ]),
        ],
        cxxLanguageStandard: .cxx20
)
