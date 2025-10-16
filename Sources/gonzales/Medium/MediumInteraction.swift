struct MediumInteraction: Interaction {

        func getDistributionModel() -> any DistributionModel {
                return phase
        }

        var dpdu = Vector()
        var faceIndex = 0
        var normal = Normal()
        var position = Point()
        var shadingNormal = Normal()
        var uv = Point2f()
        var wo = Vector()

        var phase: any PhaseFunction = HenyeyGreenstein()
}
