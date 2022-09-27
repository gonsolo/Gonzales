/// A type that can be positioned in the world via Euclidean transformations.

protocol Transformable {
        var objectToWorld: Transform { get }
        var worldToObject: Transform { get }
}

extension Transformable {
        var worldToObject: Transform {
                return objectToWorld.inverse
        }
}
