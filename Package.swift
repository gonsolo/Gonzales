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
                .systemLibrary(name: "openimageio", pkgConfig: "OpenImageIO"),
                .systemLibrary(name: "ptex", pkgConfig: "ptex"),
        ],
        swiftLanguageModes: [.v6],
        cxxLanguageStandard: .cxx20
)
