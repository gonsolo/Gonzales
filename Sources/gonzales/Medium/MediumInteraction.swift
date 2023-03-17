struct MediumInteraction: Interaction {

        func spawnRay(to: Point) -> (ray: Ray, tHit: FloatX) {
                unimplemented()
        }

        func spawnRay(inDirection direction: Vector) -> Ray {
                unimplemented()
        }

        var dpdu = Vector()
        var faceIndex = 0
        var normal = Normal()
        var position = Point()
        var shadingNormal = Normal()
        var uv = Point2F()
        var wo = Vector()
}
