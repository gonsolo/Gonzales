// swift-tools-version:5.9

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
                                "embree4",
                                "openImageIOBridge",
                                //"cuda",
                                //"cudaBridge",
                                "ptexBridge",
                        ]
                ),
                .target(
                        name: "openImageIOBridge",
                        dependencies: ["openimageio"],
                        swiftSettings: [.interoperabilityMode(.Cxx)]
                ),
                .target(
                        name: "ptexBridge",
                        dependencies: ["ptex"]
                ),
                //.target(
                //        name: "cudaBridge",
                //        dependencies: ["cuda"],
                //        swiftSettings: [.interoperabilityMode(.Cxx)]
                //),
                .systemLibrary(name: "embree4"),
                .systemLibrary(name: "openimageio", pkgConfig: "OpenImageIO"),
                //.systemLibrary(name: "cuda", pkgConfig: "cuda"),
                .systemLibrary(name: "ptex", pkgConfig: "ptex"),
        ],
        cxxLanguageStandard: .cxx20
)
