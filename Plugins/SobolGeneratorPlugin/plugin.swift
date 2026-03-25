import PackagePlugin

@main
struct SobolGeneratorPlugin: BuildToolPlugin {
    func createBuildCommands(context: PluginContext, target _: Target) async throws -> [Command] {
        // Find the executable
        let generatorExe = try context.tool(named: "SobolGenerator")

        // Input path (the direction numbers file)
        let directionNumbersPath = context.package.directoryURL
            .appendingPathComponent("Sources")
            .appendingPathComponent("libgonzales")
            .appendingPathComponent("Resources")
            .appendingPathComponent("new-joe-kuo-6.21201")

        // Output path (generated source file)
        let outputPath = context.pluginWorkDirectoryURL.appendingPathComponent("SobolMatrices.swift")

        // Create build command
        return [
            .buildCommand(
                displayName: "Generating Sobol matrices from direction numbers",
                executable: generatorExe.url,
                arguments: [
                    directionNumbersPath.path,
                    outputPath.path
                ],
                inputFiles: [directionNumbersPath],
                outputFiles: [outputPath]
            )
        ]
    }
}
