///        A material provides the look of a surface.
protocol Material {
        func computeScatteringFunctions(interaction: Interaction) -> BSDF
}
