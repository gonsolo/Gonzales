import embree

final class Embree: Accelerator {

        init(primitives: inout [Boundable & Intersectable]) {
                embreeInit()
                for primitive in primitives {
                        if let geometricPrimitive = primitive as? GeometricPrimitive {
                                if let triangle = geometricPrimitive.shape as? Triangle {
                                        geometry(triangle: triangle)
                                        bounds = union(first: bounds, second: triangle.worldBound())
                                        materials.append(geometricPrimitive.material)
                                }
                        }
                }
                embreeCommit()
        }

        deinit {
                embreeDeinit()
        }

        func commit() {
                embreeCommit()
        }

        func geometry(triangle: Triangle) {
                let points = triangle.getLocalPoints()
                let a = points.0
                let b = points.1
                let c = points.2
                embreeGeometry(a.x, a.y, a.z, b.x, b.y, b.z, c.x, c.y, c.z)
        }

        func intersect(
                ray: Ray,
                tHit: inout FloatX,
                material: MaterialIndex,
                interaction: inout SurfaceInteraction
        ) {
                var nx: FloatX = 0
                var ny: FloatX = 0
                var nz: FloatX = 0
                var tout: FloatX = 0
                var geomID: Int64 = 0
                let intersected = embreeIntersect(
                        ray.origin.x, ray.origin.y, ray.origin.z,
                        ray.direction.x, ray.direction.y, ray.direction.z,
                        0.0, tHit, &nx, &ny, &nz, &tout, &geomID)
                guard intersected else {
                        return
                }
                tHit = tout
                interaction.valid = true
                interaction.position = ray.origin + tout * ray.direction
                interaction.normal = normalized(Normal(x: nx, y: ny, z: nz))
                interaction.shadingNormal = interaction.normal
                interaction.wo = -ray.direction
                interaction.dpdu = up  // TODO
                interaction.faceIndex = 0  // TODO
                interaction.material = materials[geomID]
        }

        func worldBound() -> Bounds3f {
                return bounds
        }

        func objectBound() -> Bounds3f {
                return bounds
        }

        var materials = [Int]()
        var bounds = Bounds3f()
}
