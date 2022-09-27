import Foundation

var boundingHierarchyNodesVisited = 0

struct Node {
        var bounds = Bounds3f()
        var count = 0
        var offset: Int = 0
        var axis: Int = 0
}

@_silgen_name("_swift_stdlib_immortalize")
func _swift_stdlib_immortalize(_ p: UnsafeMutableRawPointer)

func immortalize(_ o: AnyObject) {
	withExtendedLifetime(o) { // not sure this is required
		_swift_stdlib_immortalize(Unmanaged.passUnretained(o).toOpaque())
	}
}

final class BoundingHierarchy: Boundable, Intersectable {

        init(primitives: [Intersectable], nodes: [Node]) {
                self.primitives = primitives
                for p in self.primitives {
                        immortalize(p)
                }
                self.nodes = nodes
        }

        func intersect(ray: Ray, tHit: inout FloatX, material: MaterialIndex) throws -> SurfaceInteraction {
                var interaction = SurfaceInteraction()
                var toVisit = 0
                var current = 0
                var nodesToVisit: [Int: Int] = [:]
                var nodesVisited = 0

                if nodes.isEmpty { return interaction }
                while true {
                        nodesVisited += 1
                        let node = nodes[current]
                        if node.bounds.intersects(ray: ray, tHit: &tHit) {
                                if node.count > 0 {
                                        for i in 0..<node.count {
                                                let primitive = primitives[node.offset + i]
                                                let tentativeInteraction =
                                                        try primitive.intersect(
                                                                ray: ray,
                                                                tHit: &tHit,
                                                                material: -1)
                                                if tentativeInteraction.valid {
                                                        interaction = tentativeInteraction
                                                }
                                        }
                                        if toVisit == 0 { break }
                                        toVisit -= 1
                                        current = nodesToVisit[toVisit]!
                                } else {
                                        if ray.direction[node.axis] < 0 {
                                                nodesToVisit[toVisit] = current + 1
                                                current = node.offset
                                        } else {
                                                nodesToVisit[toVisit] = node.offset
                                                current = current + 1
                                        }
                                        toVisit += 1
                                }
                        } else {
                                if toVisit == 0 { break }
                                toVisit -= 1
                                current = nodesToVisit[toVisit]!
                        }
                }
                boundingHierarchyNodesVisited += nodesVisited
                return interaction
        }

        func objectBound() -> Bounds3f {
                return worldBound()
        }

        func worldBound() -> Bounds3f {
                if nodes.isEmpty {
                        return Bounds3f()
                } else {
                        return nodes[0].bounds
                }
        }

        static func statistics() {
                print("    Nodes visited:\t\t\t\t\t\t\t\(boundingHierarchyNodesVisited)")
        }

        let primitives: [Intersectable]
        let nodes: [Node]
}
