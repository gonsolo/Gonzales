///        A material provides the look of a surface.
protocol Material {
        func setBsdf(interaction: inout SurfaceInteraction)
}
