/// A type that can be positioned in the world via Euclidean transformations.

protocol Transformable: Sendable {
        var objectToWorld: Transform { get async }
        var worldToObject: Transform { get async }
}

extension Transformable {
        var worldToObject: Transform {
                get async {
                        return await objectToWorld.inverse
                }
        }
}
