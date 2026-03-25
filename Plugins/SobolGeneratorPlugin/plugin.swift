import PackagePlugin

@main
struct SobolGeneratorPlugin: BuildToolPlugin {
    func createBuildCommands(context: PluginContext, target: Target) async throws -> [Command] {
        // Find the executable
        let generatorExe = try context.tool(named: "SobolGenerator")
        
        // Input path (the direction numbers file)
        let directionNumbersPath = context.package.directory
            .appending("Sources")
            .appending("libgonzales")
            .appending("Resources")
            .appending("new-joe-kuo-6.21201")
        
        // Output path (generated source file)
        let outputPath = context.pluginWorkDirectory.appending("SobolMatrices.swift")
        
        // Create build command
        return [
            .buildCommand(
                displayName: "Generating Sobol matrices from direction numbers",
                executable: generatorExe.path,
                arguments: [
                    directionNumbersPath.string,
                    outputPath.string
                ],
                inputFiles: [directionNumbersPath],
                outputFiles: [outputPath]
            )
        ]
    }
}
