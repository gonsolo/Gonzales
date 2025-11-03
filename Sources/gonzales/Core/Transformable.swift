/// A type that can be positioned in the world via Euclidean transformations.

protocol Transformable: Sendable {
        func getObjectToWorld(scene: Scene) -> Transform
        func getWorldToObject(scene: Scene) -> Transform
}

extension Transformable {
        func getWorldToObject(scene: Scene) -> Transform {
                return getObjectToWorld(scene: scene).inverse
        }
}
