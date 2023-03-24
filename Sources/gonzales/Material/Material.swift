///        A material provides the look of a surface.
protocol Material {
        func getBSDF(interaction: Interaction) -> BSDF
}
