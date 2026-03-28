// swift-tools-version:6.3

import PackageDescription
import CompilerPluginSupport

let package = Package(
        name: "gonzales",
        platforms: [
                .macOS(.v26)
        ],
        dependencies: [
                .package(url: "https://github.com/apple/swift-syntax.git", from: "602.0.0")
        ],
        targets: [
                .macro(
                        name: "DevirtualizeMacroPlugin",
                        dependencies: [
                                .product(name: "SwiftSyntaxMacros", package: "swift-syntax"),
                                .product(name: "SwiftCompilerPlugin", package: "swift-syntax")
                        ]
                ),
                .target(
                        name: "DevirtualizeMacro",
                        dependencies: ["DevirtualizeMacroPlugin"]
                ),
                .target(
                        name: "libgonzales",
                        dependencies: [
                                "openImageIOBridge",
                                "ptexBridge",
                                "DevirtualizeMacro",
                                "mojoKernel",
                                "openimagedenoise",
                                "vulkanViewer",
                        ],
                        resources: [
                                .copy("Resources/new-joe-kuo-6.21201")
                        ],
                        swiftSettings: [
                                .unsafeFlags(["-Ounchecked"]),
                        ],
                        plugins: ["SobolGeneratorPlugin"]
                ),
                .testTarget(
                        name: "libgonzalesTests",
                        dependencies: [
                                "libgonzales"
                        ],
                        swiftSettings: [
                                .interoperabilityMode(.Cxx),
                                .unsafeFlags(["-Ounchecked"]),
                        ],
                        linkerSettings: [
                                .unsafeFlags(["-L.build", "-lmojo"])
                        ]
                ),
                .executableTarget(
                        name: "gonzales",
                        dependencies: [
                                "libgonzales",
                        ],
                        swiftSettings: [
                                .unsafeFlags(["-Ounchecked"]),
                        ],
                        linkerSettings: [
                                .unsafeFlags(["-L.build", "-lmojo"])
                        ]
                ),
                .executableTarget(
                        name: "SobolGenerator"
                ),
                .plugin(
                        name: "SobolGeneratorPlugin",
                        capability: .buildTool(),
                        dependencies: ["SobolGenerator"]
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
                .target(
                        name: "mojoKernel",
                        path: "Sources/mojoKernel",
                        publicHeadersPath: "include"
                ),
                .systemLibrary(name: "openimageio", pkgConfig: "OpenImageIO"),
                .systemLibrary(name: "ptex", pkgConfig: "ptex"),
                .systemLibrary(name: "openimagedenoise"),
                .target(
                        name: "vulkanViewer",
                        path: "Sources/vulkanViewer",
                        exclude: ["shaders", "generated"],
                        publicHeadersPath: "include",
                        cxxSettings: [
                                .headerSearchPath("generated")
                        ],
                        linkerSettings: [
                                .linkedLibrary("vulkan"),
                                .linkedLibrary("glfw")
                        ]
                ),
        ],
        swiftLanguageModes: [.v6],
        cxxLanguageStandard: .cxx20
)
