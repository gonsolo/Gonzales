@MainActor
var triangleIntersections = 0

@MainActor
var triangleHits = 0

@MainActor
var numberOfTriangles = 0

@MainActor
var triangleMemory = 0

@MainActor
var worldBoundCalled = 0

struct TriangleMesh {

        init() {
                objectToWorld = Transform()
                numberTriangles = 0
                vertexIndices = []
                points = []
                normals = []
                uvs = []
                faceIndices = []
        }

        init(
                objectToWorld: Transform,
                numberTriangles: Int,
                vertexIndices: [Int],
                points: [Point],
                normals: [Normal],
                uvs: [Vector2F],
                faceIndices: [Int]
        ) {
                self.objectToWorld = objectToWorld
                self.vertexIndices = vertexIndices
                self.numberTriangles = numberTriangles
                self.points = points
                self.normals = normals
                self.uvs = uvs
                self.faceIndices = faceIndices
        }

        func getVertexIndex(at index: Int) -> Int {
                vertexIndices[index]
        }

        func getPoint(at index: Int) -> Point {
                points[index]
        }

        func getUVs(indices: (Int, Int, Int)) -> (Vector2F, Vector2F, Vector2F) {
                guard !uvs.isEmpty else {
                        return (Vector2F(), Vector2F(), Vector2F())
                }
                return (
                        Vector2F(uvs[indices.0]),
                        Vector2F(uvs[indices.1]),
                        Vector2F(uvs[indices.2])
                )
        }

        func getObjectToWorld() -> Transform {
                objectToWorld
        }

        var pointCount: Int {
                points.count
        }

        let numberTriangles: Int
        let normals: [Normal]
        let faceIndices: [Int]

        let objectToWorld: Transform
        private let points: [Point]
        private let vertexIndices: [Int]
        private let uvs: [Vector2F]
}

enum TriangleError: Error {
        case index
        case vertexIndices
}

extension Array {

        subscript<Index: BinaryInteger>(i: Index) -> Element {
                get {
                        return self[Int(i)]
                }
        }
}

final class TriangleMeshBuilder {

        func appendMesh(mesh: TriangleMesh) -> Int {
                var meshIndex = 0
                meshIndex = meshes.count
                meshes.append(mesh)
                return meshIndex
        }

        func getMeshes() -> TriangleMeshes {
                return TriangleMeshes(meshes: meshes)
        }

        private var meshes: [TriangleMesh] = []
}

struct TriangleMeshes {

        func getMesh(index: Int) -> TriangleMesh {
                return meshes[index]
        }

        func getUVFor(meshIndex: Int, indices: (Int, Int, Int)) -> (Vector2F, Vector2F, Vector2F) {
                return meshes[meshIndex].getUVs(indices: indices)
        }

        func getPointCountFor(meshIndex: Int) -> Int {
                return meshes[meshIndex].pointCount
        }

        func getVertexIndexFor(meshIndex: Int, at vertexIndex: Int) -> Int {
                return meshes[meshIndex].getVertexIndex(at: vertexIndex)
        }

        func getPointFor(meshIndex: Int, at vertexIndex: Int) -> Point {
                return meshes[meshIndex].getPoint(at: vertexIndex)
        }

        func getNormal(meshIndex: Int, vertexIndex: Int) -> Normal {
                return meshes[meshIndex].normals[vertexIndex]
        }

        func hasNormals(meshIndex: Int) -> Bool {
                return !meshes[meshIndex].normals.isEmpty
        }

        func hasFaceIndices(meshIndex: Int) -> Bool {
                return !meshes[meshIndex].faceIndices.isEmpty
        }

        func getFaceIndex(meshIndex: Int, index: Int) -> Int {
                return meshes[meshIndex].faceIndices[index]
        }

        func getObjectToWorldFor(meshIndex: Int) -> Transform {
                return meshes[meshIndex].objectToWorld
        }

        let meshes: [TriangleMesh]
}

@MainActor
var triangleMeshBuilder = TriangleMeshBuilder()

struct TriangleIntersection {
        init() {
                primId = PrimId()
                t = FloatX.greatestFiniteMagnitude
        }

        init(primId: PrimId, t: FloatX) {
                self.primId = primId
                self.t = t
        }
        let primId: PrimId
        let t: FloatX
}

struct TriangleIntersectionFull {
        let t: FloatX
        let b0: FloatX
        let b1: FloatX
        let b2: FloatX
        let pHit: Point?  // Optional, only needed for SurfaceInteraction, but calculate outside the test for efficiency
        let dp02: Vector
        let dp12: Vector
}

struct Triangle: Shape {
        typealias IntersectionData = TriangleIntersection

        init(
                meshIndex: Int,
                number: Int,
                triangleMeshes: TriangleMeshes
        ) throws {
                self.meshIndex = meshIndex
                self.idx = 3 * number

                //numberOfTriangles += 1
                //triangleMemory += MemoryLayout<Self>.stride
                let pointCount = triangleMeshes.getPointCountFor(meshIndex: meshIndex)
                guard
                        triangleMeshes.getVertexIndexFor(meshIndex: meshIndex, at: idx + 0) < pointCount
                                && triangleMeshes.getVertexIndexFor(meshIndex: meshIndex, at: idx + 1)
                                        < pointCount
                                && triangleMeshes.getVertexIndexFor(meshIndex: meshIndex, at: idx + 2)
                                        < pointCount
                else {
                        throw TriangleError.index
                }
        }

        static func formatHuman(_ number: Int) -> String {
                if number > 1024 * 1024 {
                        return String(format: "%.1f", Double(number) / 1024.0 / 1024.0) + "MB"
                }
                if number > 1024 {
                        return String(number / 1024) + "KB"
                }
                return String(number) + " bytes"
        }

        @MainActor
        static func statistics() {
                print("  Number of triangles:\t\t\t\t\t\t\t\(numberOfTriangles)")
                print("  Triangle memory:\t\t\t\t\t\t\t\(formatHuman(triangleMemory))")
                let ratio = Float(triangleHits) / Float(triangleIntersections)
                let intersectionRatio = String(format: " (%.2f)", ratio)
                print("  Ray-Triangle intersection tests:\t\t\t\t", terminator: "")
                print("\(triangleHits) /\t\(triangleIntersections)\(intersectionRatio)")
                print("  Triangle worldBound calls:\t\t\t\t\t\t\(worldBoundCalled)")
        }

        var triangleMeshes: TriangleMeshes {
                return scene.meshes
        }

        var vertexIndex0: Int {
                return triangleMeshes.getVertexIndexFor(meshIndex: meshIndex, at: idx + 0)
        }

        var vertexIndex1: Int {
                return triangleMeshes.getVertexIndexFor(meshIndex: meshIndex, at: idx + 1)
        }

        var vertexIndex2: Int {
                return triangleMeshes.getVertexIndexFor(meshIndex: meshIndex, at: idx + 2)
        }

        var point0: Point {
                return triangleMeshes.getPointFor(meshIndex: meshIndex, at: vertexIndex0)
        }

        var point1: Point {
                return triangleMeshes.getPointFor(meshIndex: meshIndex, at: vertexIndex1)
        }

        var point2: Point {
                return triangleMeshes.getPointFor(meshIndex: meshIndex, at: vertexIndex2)
        }

        func objectBound() -> Bounds3f {
                let (p0, p1, p2) = getLocalPoints()
                return union(bound: Bounds3f(first: p0, second: p1), point: p2)
        }

        func worldBound() -> Bounds3f {
                //worldBoundCalled += 1
                return objectToWorld * objectBound()
        }

        func computeUVHit(b0: FloatX, b1: FloatX, b2: FloatX, uv: (Vector2F, Vector2F, Vector2F))
                -> Point2f
        {
                let uvHit0: Point2f = b0 * Point2f(from: uv.0)
                let uvHit1: Point2f = b1 * Point2f(from: uv.1)
                let uvHit2: Point2f = b2 * Point2f(from: uv.2)
                let uvHit: Point2f = uvHit0 + uvHit1 + uvHit2
                return uvHit
        }

        func getIntersectionDataFull(
                ray worldRay: Ray,
                tHit: inout FloatX
        ) throws -> TriangleIntersectionFull? {

                // Transform the ray to object space
                let ray = objectToWorld * worldRay

                // --- Setup and Plane Projection ---

                var p0t: Point = point0 - ray.origin
                var p1t: Point = point1 - ray.origin
                var p2t: Point = point2 - ray.origin

                let kz = maxDimension(abs(ray.direction))
                let kx = (kz + 1) % 3
                let ky = (kx + 1) % 3
                let d: Vector = permute(vector: ray.direction, x: kx, y: ky, z: kz)
                p0t = permute(point: p0t, x: kx, y: ky, z: kz)
                p1t = permute(point: p1t, x: kx, y: ky, z: kz)
                p2t = permute(point: p2t, x: kx, y: ky, z: kz)

                let sx: FloatX = -d.x / d.z
                let sy: FloatX = -d.y / d.z
                let sz: FloatX = 1.0 / d.z

                // Shearing transformation
                p0t.x += sx * p0t.z
                p0t.y += sy * p0t.z
                p1t.x += sx * p1t.z
                p1t.y += sy * p1t.z
                p2t.x += sx * p2t.z
                p2t.y += sy * p2t.z

                // Compute edge functions e0, e1, e2
                let e0: FloatX = p1t.x * p2t.y - p1t.y * p2t.x
                let e1: FloatX = p2t.x * p0t.y - p2t.y * p0t.x
                let e2: FloatX = p0t.x * p1t.y - p0t.y * p1t.x

                // Check edge functions for hit (same sign)
                if (e0 < 0 || e1 < 0 || e2 < 0) && (e0 > 0 || e1 > 0 || e2 > 0) {
                        return nil
                }
                let det: FloatX = e0 + e1 + e2
                if det == 0 {
                        return nil  // Degenerate triangle or ray parallel to plane
                }

                // --- Compute t value and check range ---

                p0t.z *= sz
                p1t.z *= sz
                p2t.z *= sz
                let tScaled: FloatX = e0 * p0t.z + e1 * p1t.z + e2 * p2t.z

                // Ray t range test against tHit and ray segment limits (0)
                let hitCondition = det > 0
                if (hitCondition && (tScaled <= 0 || tScaled > tHit * det))
                        || (!hitCondition && (tScaled >= 0 || tScaled < tHit * det))
                {
                        return nil
                }

                // --- Intersection found ---

                let invDet: FloatX = 1 / det
                let b0: FloatX = e0 * invDet
                let b1: FloatX = e1 * invDet
                let b2: FloatX = e2 * invDet
                let t: FloatX = tScaled * invDet

                tHit = t  // Update closest hit distance

                // Calculate necessary geometric data
                let dp02 = Vector(point: point0 - point2)
                let dp12 = Vector(point: point1 - point2)

                return TriangleIntersectionFull(
                        t: t,
                        b0: b0,
                        b1: b1,
                        b2: b2,
                        pHit: nil,  // pHit calculation is only needed for SurfaceInteraction
                        dp02: dp02,
                        dp12: dp12
                )
        }

        func getIntersectionData(
                ray worldRay: Ray,
                tHit: inout FloatX,
                data: inout TriangleIntersection
        ) throws -> Bool {

                // Transform the ray to object space
                let ray = objectToWorld * worldRay

                // --- Setup and Plane Projection ---

                var p0t: Point = point0 - ray.origin
                var p1t: Point = point1 - ray.origin
                var p2t: Point = point2 - ray.origin

                let kz = maxDimension(abs(ray.direction))
                let kx = (kz + 1) % 3
                let ky = (kx + 1) % 3
                let d: Vector = permute(vector: ray.direction, x: kx, y: ky, z: kz)
                p0t = permute(point: p0t, x: kx, y: ky, z: kz)
                p1t = permute(point: p1t, x: kx, y: ky, z: kz)
                p2t = permute(point: p2t, x: kx, y: ky, z: kz)

                let sx: FloatX = -d.x / d.z
                let sy: FloatX = -d.y / d.z
                let sz: FloatX = 1.0 / d.z

                // Shearing transformation
                p0t.x += sx * p0t.z
                p0t.y += sy * p0t.z
                p1t.x += sx * p1t.z
                p1t.y += sy * p1t.z
                p2t.x += sx * p2t.z
                p2t.y += sy * p2t.z

                // Compute edge functions e0, e1, e2
                let e0: FloatX = p1t.x * p2t.y - p1t.y * p2t.x
                let e1: FloatX = p2t.x * p0t.y - p2t.y * p0t.x
                let e2: FloatX = p0t.x * p1t.y - p0t.y * p1t.x

                // Check edge functions for hit (same sign)
                if (e0 < 0 || e1 < 0 || e2 < 0) && (e0 > 0 || e1 > 0 || e2 > 0) {
                        return false
                }
                let det: FloatX = e0 + e1 + e2
                if det == 0 {
                        return false // Degenerate triangle or ray parallel to plane
                }

                // --- Compute t value and check range ---

                p0t.z *= sz
                p1t.z *= sz
                p2t.z *= sz
                let tScaled: FloatX = e0 * p0t.z + e1 * p1t.z + e2 * p2t.z

                // Ray t range test against tHit and ray segment limits (0)
                let hitCondition = det > 0
                if (hitCondition && (tScaled <= 0 || tScaled > tHit * det))
                        || (!hitCondition && (tScaled >= 0 || tScaled < tHit * det))
                {
                        return false
                }

                // --- Intersection found ---

                let invDet: FloatX = 1 / det
                //let b0: FloatX = e0 * invDet
                //let b1: FloatX = e1 * invDet
                //let b2: FloatX = e2 * invDet
                let t: FloatX = tScaled * invDet

                tHit = t  // Update closest hit distance

                // Calculate necessary geometric data
                //let dp02 = Vector(point: point0 - point2)
                //let dp12 = Vector(point: point1 - point2)

                data = TriangleIntersection(
                        primId: PrimId(id1: meshIndex, id2: idx, type: .triangle),
                        t: t,
                )
                return true
        }

        func intersect(
                ray worldRay: Ray,
                tHit: inout FloatX
        ) throws -> Bool {
                var notUsed = TriangleIntersection()
                return try getIntersectionData(ray: worldRay, tHit: &tHit, data: &notUsed)
        }

        func computeSurfaceInteraction(
                data: TriangleIntersection,
                worldRay: Ray,
                interaction: inout SurfaceInteraction
        ) {
                var varT = data.t
                var data: TriangleIntersectionFull?
                do {
                        data = try getIntersectionDataFull(ray: worldRay, tHit: &varT)
                } catch {
                        fatalError("getIntersectionDataFull in computeSurfaceInteraction!")
                }
                guard let data = data else {
                       return
                }
                // --- Calculate Hit Point (pHit) ---
                let hit0: Point = data.b0 * point0
                let hit1: Point = data.b1 * point1
                let hit2: Point = data.b2 * point2
                let pHit: Point = hit0 + hit1 + hit2

                // --- Geometric Normal ---
                let normal = normalized(Normal(cross(data.dp02, data.dp12)))

                // --- UVs, Tangent Space (dpdu/dpdv), and Shading Normal ---

                let uv = triangleMeshes.getUVFor(
                        meshIndex: meshIndex,
                        indices: (vertexIndex0, vertexIndex1, vertexIndex2))
                let uvHit = computeUVHit(b0: data.b0, b1: data.b1, b2: data.b2, uv: uv)

                let duv02: Vector2F = uv.0 - uv.2
                let duv12: Vector2F = uv.1 - uv.2
                let determinantUV: FloatX = duv02[0] * duv12[1] - duv02[1] * duv12[0]
                let degenerateUV: Bool = abs(determinantUV) < 1e-8
                var dpdu = up  // Assuming 'up' is a predefined Vector (e.g., Vector(0,0,0) or local Y axis)
                var dpdv = up

                if !degenerateUV {
                        let invDeterminantUV = 1 / determinantUV
                        let a: Vector = +duv12[1] * data.dp02
                        let b: Vector = +duv02[1] * data.dp12
                        let c: Vector = -duv12[0] * data.dp02
                        let d: Vector = +duv02[0] * data.dp12
                        dpdu = (a - b) * invDeterminantUV
                        dpdv = (c - d) * invDeterminantUV
                }

                if degenerateUV || lengthSquared(cross(dpdu, dpdv)) == 0 {
                        let ng: Vector = cross(point2 - point0, point1 - point0)
                        if lengthSquared(ng) == 0 {
                                return  // Cannot compute valid normal/tangent space
                        }
                        (dpdu, dpdv) = makeCoordinateSystem(from: normalized(ng))
                }

                var shadingNormal: Normal
                if !triangleMeshes.hasNormals(meshIndex: meshIndex) {
                        shadingNormal = normal
                } else {
                        let n0 = triangleMeshes.getNormal(
                                meshIndex: meshIndex, vertexIndex: vertexIndex0)
                        let sn0 = data.b0 * n0
                        let n1 = triangleMeshes.getNormal(
                                meshIndex: meshIndex, vertexIndex: vertexIndex1)
                        let sn1 = data.b1 * n1
                        let n2 = triangleMeshes.getNormal(
                                meshIndex: meshIndex, vertexIndex: vertexIndex2)
                        let sn2 = data.b2 * n2
                        shadingNormal = sn0 + sn1 + sn2
                        if lengthSquared(shadingNormal) > 0 {
                                shadingNormal = normalized(shadingNormal)
                        } else {
                                shadingNormal = Normal(x: 6, y: 6, z: 6)  // Fallback as in original
                        }
                }

                var ss = normalized(dpdu)
                var ts = cross(ss, Vector(normal: shadingNormal))
                if lengthSquared(ts) > 0 {
                        ts.normalize()
                        ss = cross(ts, Vector(normal: shadingNormal))
                } else {
                        (ss, ts) = makeCoordinateSystem(from: Vector(normal: shadingNormal))
                }

                // Set shading geometry
                dpdu = ss

                var faceIndex: Int = 0
                if triangleMeshes.hasFaceIndices(meshIndex: meshIndex) {
                        faceIndex = triangleMeshes.getFaceIndex(
                                meshIndex: meshIndex,
                                index: idx / 3)
                }

                // --- Finalize SurfaceInteraction ---
                let rayObjectSpace = objectToWorld * worldRay

                interaction.valid = true
                interaction.position = objectToWorld * pHit
                interaction.normal = normalized(objectToWorld * normal)
                interaction.shadingNormal = normalized(objectToWorld * shadingNormal)
                interaction.wo = normalized(objectToWorld * -rayObjectSpace.direction)
                interaction.dpdu = dpdu
                interaction.uv = uvHit
                interaction.faceIndex = faceIndex
        }

        func intersect(
                ray worldRay: Ray,
                tHit: inout FloatX,
                interaction: inout SurfaceInteraction
        ) throws {
                var data = TriangleIntersection()
                if try !getIntersectionData(ray: worldRay, tHit: &tHit, data: &data) {
                        return  
                }

                // 2. Compute the full SurfaceInteraction using the new private method
                computeSurfaceInteraction(
                        data: data,
                        worldRay: worldRay,
                        interaction: &interaction
                )
        }

        private func getLocalPoint(index: Int) -> Point {
                return triangleMeshes.getPointFor(meshIndex: meshIndex, at: index)
        }

        private func getWorldPoint(index: Int) -> Point {
                return objectToWorld * getLocalPoint(index: index)
        }

        public func getLocalPoints() -> (Point, Point, Point) {
                let p0 = getLocalPoint(index: vertexIndex0)
                let p1 = getLocalPoint(index: vertexIndex1)
                let p2 = getLocalPoint(index: vertexIndex2)
                return (p0, p1, p2)
        }

        public func getWorldPoints() -> (Point, Point, Point) {
                let p0 = getWorldPoint(index: vertexIndex0)
                let p1 = getWorldPoint(index: vertexIndex1)
                let p2 = getWorldPoint(index: vertexIndex2)
                return (p0, p1, p2)
        }

        private func uniformSampleTriangle(u: TwoRandomVariables) -> Point2f {
                let su0 = u.0.squareRoot()
                return Point2f(x: 1 - su0, y: u.1 * su0)
        }

        func area() -> FloatX {
                let (p0, p1, p2) = getLocalPoints()
                return 0.5 * length(cross(Vector(vector: (p1 - p0)), p2 - p0))
        }

        func sample(u: TwoRandomVariables) -> (interaction: SurfaceInteraction, pdf: FloatX) {
                let b = uniformSampleTriangle(u: u)
                let (p0, p1, p2) = getLocalPoints()
                let sampled0: Point = b[0] * p0
                let sampled1: Point = b[1] * p1
                let sampled2: Point = (1 - b[0] - b[1]) * p2
                let localPoint: Point = sampled0 + sampled1 + sampled2
                let localNormal = normalized(Normal(cross(p1 - p0, p2 - p0)))
                let worldPoint = objectToWorld * localPoint
                let worldNormal = objectToWorld * localNormal
                let worldInteraction = SurfaceInteraction(position: worldPoint, normal: worldNormal)
                let pdf = 1 / area()
                return (worldInteraction, pdf)
        }

        @MainActor
        var description: String {
                var d = "Triangle [ "
                //let (p0, p1, p2) = getLocalPoints()
                //d += p0.description + p1.description + p2.description
                d += " ]"
                return d
        }

        var objectToWorld: Transform {
                return triangleMeshes.getObjectToWorldFor(meshIndex: meshIndex)
        }

        let meshIndex: Int
        let idx: Int
        //let triangleMeshes: TriangleMeshes
}

@MainActor
func createTriangleMeshShape(
        objectToWorld: Transform,
        parameters: ParameterDictionary
) throws -> [ShapeType] {
        let indices = try parameters.findInts(name: "indices")
        guard indices.count % 3 == 0 else {
                throw ApiError.input(message: "Triangle indices must be multiplies of 3")
        }
        let points = try parameters.findPoints(name: "P")
        let normals = try parameters.findNormals(name: "N")
        let uvFloatXs = try parameters.findFloatXs(called: "uv")
        var uvs = [Vector2F]()
        for i in 0..<uvFloatXs.count / 2 {
                uvs.append(Vector2F(x: uvFloatXs[2 * i], y: uvFloatXs[2 * i + 1]))
        }
        let faceIndices = try parameters.findInts(name: "faceIndices")

        return try createTriangleMesh(
                objectToWorld: objectToWorld,
                indices: indices,
                points: points,
                normals: normals,
                uvs: uvs,
                faceIndices: faceIndices)
}

@MainActor
func createTriangleMesh(
        objectToWorld: Transform,
        indices: [Int],
        points: [Point],
        normals: [Normal],
        uvs: [Vector2F],
        faceIndices: [Int]
) throws -> [ShapeType] {
        let numberTriangles = indices.count / 3
        let trianglePoints = points
        let triangleNormals = normals
        let triangleUvs = uvs
        var triangles = [ShapeType]()

        let mesh = TriangleMesh(
                objectToWorld: objectToWorld,
                numberTriangles: numberTriangles,
                vertexIndices: indices,
                points: trianglePoints,
                normals: triangleNormals,
                uvs: triangleUvs,
                faceIndices: faceIndices)

        let meshIndex = triangleMeshBuilder.appendMesh(mesh: mesh)
        let triangleMeshes = triangleMeshBuilder.getMeshes()

        for i in 0..<numberTriangles {
                triangles.append(
                        try .triangle(
                                Triangle(
                                        meshIndex: meshIndex,
                                        number: i,
                                        triangleMeshes: triangleMeshes)))
        }
        return triangles
}
