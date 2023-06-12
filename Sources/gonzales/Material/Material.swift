///        A material provides the look of a surface.
protocol Material {
        func getGlobalBsdf(interaction: Interaction) -> GlobalBsdf
}
