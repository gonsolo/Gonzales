import Foundation
import mojoKernel
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
                self.points = points.map { objectToWorld * $0 }
                self.normals = normals.map { objectToWorld * $0 }
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

        var cStruct: TriangleMesh_C {
                let pPtr = points.withUnsafeBufferPointer { $0.baseAddress! }
                let fPtr = faceIndices.withUnsafeBufferPointer { $0.baseAddress! }
                let vPtr = vertexIndices.withUnsafeBufferPointer { $0.baseAddress! }

                return TriangleMesh_C(
                        points: UnsafeRawPointer(pPtr).assumingMemoryBound(to: Float.self),
                        faceIndices: UnsafeRawPointer(fPtr).assumingMemoryBound(to: Int64.self),
                        vertexIndices: UnsafeRawPointer(vPtr).assumingMemoryBound(to: Int64.self)
                )
        }
}

enum TriangleError: Error {
        case index
        case vertexIndices
}

extension Array {

        subscript<Index: BinaryInteger>(index: Index) -> Element {
                return self[Int(index)]
        }
}

final class TriangleMeshBuilder: @unchecked Sendable {
        private let lock = NSLock()

        func appendMesh(mesh: TriangleMesh) -> Int {
                lock.lock()
                defer { lock.unlock() }
                let meshIndex = meshes.count
                meshes.append(mesh)
                return meshIndex
        }

        func getMeshes() -> TriangleMeshes {
                lock.lock()
                defer { lock.unlock() }
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

struct TriangleIntersection {
        init() {
                primId = PrimId()
                tValue = Real.greatestFiniteMagnitude
                barycentric0 = 0
                barycentric1 = 0
                barycentric2 = 0
        }

        init(primId: PrimId, tValue: Real, barycentric0: Real, barycentric1: Real, barycentric2: Real) {
                self.primId = primId
                self.tValue = tValue
                self.barycentric0 = barycentric0
                self.barycentric1 = barycentric1
                self.barycentric2 = barycentric2
        }
        let primId: PrimId
        let tValue: Real
        let barycentric0: Real
        let barycentric1: Real
        let barycentric2: Real
}

struct Triangle: Shape {
        let meshIndex: Int
        let triangleIndex: Int

        init(
                meshIndex: Int,
                number: Int
        ) {
                self.meshIndex = meshIndex
                self.triangleIndex = 3 * number
        }

}

extension Triangle {
        static func formatHuman(_ number: Int) -> String {
                if number > 1024 * 1024 {
                        return String(format: "%.1f", Double(number) / 1024.0 / 1024.0) + "MB"
                }
                if number > 1024 {
                        return String(number / 1024) + "KB"
                }
                return String(number) + " bytes"
        }

        func getTriangleMeshes(scene: Scene) -> TriangleMeshes {
                return scene.meshes
        }

        func getVertexIndex0(scene: Scene) -> Int {
                return getTriangleMeshes(scene: scene).getVertexIndexFor(
                        meshIndex: meshIndex, at: triangleIndex + 0)
        }

        func getVertexIndex1(scene: Scene) -> Int {
                return getTriangleMeshes(scene: scene).getVertexIndexFor(
                        meshIndex: meshIndex, at: triangleIndex + 1)
        }

        func getVertexIndex2(scene: Scene) -> Int {
                return getTriangleMeshes(scene: scene).getVertexIndexFor(
                        meshIndex: meshIndex, at: triangleIndex + 2)
        }

        func getPoint0(scene: Scene) -> Point {
                return getTriangleMeshes(scene: scene).getPointFor(
                        meshIndex: meshIndex, at: getVertexIndex0(scene: scene))
        }

        func getPoint1(scene: Scene) -> Point {
                return getTriangleMeshes(scene: scene).getPointFor(
                        meshIndex: meshIndex, at: getVertexIndex1(scene: scene))
        }

        func getPoint2(scene: Scene) -> Point {
                return getTriangleMeshes(scene: scene).getPointFor(
                        meshIndex: meshIndex, at: getVertexIndex2(scene: scene))
        }

        func objectBound(scene: Scene) -> Bounds3f {
                let (point0, point1, point2) = getLocalPoints(scene: scene)
                return union(bound: Bounds3f(first: point0, second: point1), point: point2)
        }

        func worldBound(scene: Scene) -> Bounds3f {
                return objectBound(scene: scene)
        }

        func computeUVHit(
                barycentric0: Real, barycentric1: Real, barycentric2: Real,
                uvCoordinates: (Vector2F, Vector2F, Vector2F)
        ) -> Point2f {
                let uvHit0: Point2f = barycentric0 * Point2f(from: uvCoordinates.0)
                let uvHit1: Point2f = barycentric1 * Point2f(from: uvCoordinates.1)
                let uvHit2: Point2f = barycentric2 * Point2f(from: uvCoordinates.2)
                let uvHit: Point2f = uvHit0 + uvHit1 + uvHit2
                return uvHit
        }

        func getIntersectionData(
                scene: Scene,
                ray worldRay: Ray,
                tHit: inout Real,
                data: inout TriangleIntersection
        ) -> Bool {

                // Ray is processed natively in world space
                let ray = worldRay

                // --- Setup and Plane Projection ---

                var p0t: Point = getPoint0(scene: scene) - ray.origin
                var p1t: Point = getPoint1(scene: scene) - ray.origin
                var p2t: Point = getPoint2(scene: scene) - ray.origin

                let axisZ = maxDimension(abs(ray.direction))
                let axisX = (axisZ + 1) % 3
                let axisY = (axisX + 1) % 3
                let directionVector: Vector = permute(vector: ray.direction, x: axisX, y: axisY, z: axisZ)
                p0t = permute(point: p0t, x: axisX, y: axisY, z: axisZ)
                p1t = permute(point: p1t, x: axisX, y: axisY, z: axisZ)
                p2t = permute(point: p2t, x: axisX, y: axisY, z: axisZ)

                let shearX: Real = -directionVector.x / directionVector.z
                let shearY: Real = -directionVector.y / directionVector.z
                let shearZ: Real = 1.0 / directionVector.z

                // Shearing transformation
                p0t.x += shearX * p0t.z
                p0t.y += shearY * p0t.z
                p1t.x += shearX * p1t.z
                p1t.y += shearY * p1t.z
                p2t.x += shearX * p2t.z
                p2t.y += shearY * p2t.z

                // Compute edge functions edge0, edge1, edge2
                let edge0: Real = p1t.x * p2t.y - p1t.y * p2t.x
                let edge1: Real = p2t.x * p0t.y - p2t.y * p0t.x
                let edge2: Real = p0t.x * p1t.y - p0t.y * p1t.x

                // Check edge functions for hit (same sign)
                if (edge0 < 0 || edge1 < 0 || edge2 < 0) && (edge0 > 0 || edge1 > 0 || edge2 > 0) {
                        return false
                }
                let det: Real = edge0 + edge1 + edge2
                if det == 0 {
                        return false  // Degenerate triangle or ray parallel to plane
                }

                // --- Compute t value and check range ---

                p0t.z *= shearZ
                p1t.z *= shearZ
                p2t.z *= shearZ
                let tScaled: Real = edge0 * p0t.z + edge1 * p1t.z + edge2 * p2t.z

                // Ray t range test against tHit and ray segment limits (0)
                let hitCondition = det > 0
                if (hitCondition && (tScaled <= 0 || tScaled > tHit * det))
                        || (!hitCondition && (tScaled >= 0 || tScaled < tHit * det)) {
                        return false
                }

                // --- Intersection found ---

                let invDet: Real = 1 / det
                let barycentric0: Real = edge0 * invDet
                let barycentric1: Real = edge1 * invDet
                let barycentric2: Real = edge2 * invDet
                let tValue: Real = tScaled * invDet

                tHit = tValue  // Update closest hit distance

                data = TriangleIntersection(
                        primId: PrimId(id1: meshIndex, id2: triangleIndex, type: .triangle),
                        tValue: tValue,
                        barycentric0: barycentric0,
                        barycentric1: barycentric1,
                        barycentric2: barycentric2
                )
                return true
        }

        func intersect(
                scene: Scene,
                ray worldRay: Ray,
                tHit: inout Real
        ) -> Bool {
                var notUsed = TriangleIntersection()
                return getIntersectionData(scene: scene, ray: worldRay, tHit: &tHit, data: &notUsed)
        }

        func computeSurfaceInteraction(
                scene: Scene,
                data: TriangleIntersection,
                worldRay: Ray
        ) -> SurfaceInteraction? {
                let barycentric0 = data.barycentric0
                let barycentric1 = data.barycentric1
                let barycentric2 = data.barycentric2

                // Calculate necessary geometric data
                let dp02 = Vector(point: getPoint0(scene: scene) - getPoint2(scene: scene))
                let dp12 = Vector(point: getPoint1(scene: scene) - getPoint2(scene: scene))

                var interaction = SurfaceInteraction()
                // --- Calculate Hit Point (pHit) ---
                let hit0: Point = barycentric0 * getPoint0(scene: scene)
                let hit1: Point = barycentric1 * getPoint1(scene: scene)
                let hit2: Point = barycentric2 * getPoint2(scene: scene)
                let pHitValue: Point = hit0 + hit1 + hit2

                // --- Geometric Normal ---
                let normal = normalized(Normal(cross(dp02, dp12)))

                // --- UVs, Tangent Space (dpdu/dpdv), and Shading Normal ---

                let uvCoordinates = getTriangleMeshes(scene: scene).getUVFor(
                        meshIndex: meshIndex,
                        indices: (
                                getVertexIndex0(scene: scene), getVertexIndex1(scene: scene),
                                getVertexIndex2(scene: scene)
                        ))
                let uvHit = computeUVHit(
                        barycentric0: barycentric0, barycentric1: barycentric1,
                        barycentric2: barycentric2, uvCoordinates: uvCoordinates)

                let duv02: Vector2F = uvCoordinates.0 - uvCoordinates.2
                let duv12: Vector2F = uvCoordinates.1 - uvCoordinates.2
                let determinantUV: Real = duv02[0] * duv12[1] - duv02[1] * duv12[0]
                let degenerateUV: Bool = abs(determinantUV) < 1e-8
                var dpdu = upVector  // Assuming 'upVector' is a predefined Vector (e.g., Vector(0,0,0) or local Y axis)
                var dpdv = upVector

                if !degenerateUV {
                        let invDeterminantUV = 1 / determinantUV
                        let termA: Vector = +duv12[1] * dp02
                        let termB: Vector = +duv02[1] * dp12
                        let termC: Vector = -duv12[0] * dp02
                        let termD: Vector = +duv02[0] * dp12
                        dpdu = (termA - termB) * invDeterminantUV
                        dpdv = (termC - termD) * invDeterminantUV
                }

                if degenerateUV || lengthSquared(cross(dpdu, dpdv)) == 0 {
                        let geometricNormal: Vector = cross(
                                getPoint2(scene: scene) - getPoint0(scene: scene),
                                getPoint1(scene: scene) - getPoint0(scene: scene))
                        if lengthSquared(geometricNormal) == 0 {
                                return nil  // Cannot compute valid normal/tangent space
                        }
                        (dpdu, dpdv) = makeCoordinateSystem(from: normalized(geometricNormal))
                }

                var shadingNormal: Normal
                if !getTriangleMeshes(scene: scene).hasNormals(meshIndex: meshIndex) {
                        shadingNormal = normal
                } else {
                        let normal0 = getTriangleMeshes(scene: scene).getNormal(
                                meshIndex: meshIndex, vertexIndex: getVertexIndex0(scene: scene))
                        let sn0 = barycentric0 * normal0
                        let normal1 = getTriangleMeshes(scene: scene).getNormal(
                                meshIndex: meshIndex, vertexIndex: getVertexIndex1(scene: scene))
                        let sn1 = barycentric1 * normal1
                        let normal2 = getTriangleMeshes(scene: scene).getNormal(
                                meshIndex: meshIndex, vertexIndex: getVertexIndex2(scene: scene))
                        let sn2 = barycentric2 * normal2
                        shadingNormal = sn0 + sn1 + sn2
                        if lengthSquared(shadingNormal) > 0 {
                                shadingNormal = normalized(shadingNormal)
                        } else {
                                shadingNormal = Normal(x: 6, y: 6, z: 6)  // Fallback as in original
                        }
                }

                var shadingS = normalized(dpdu)
                var shadingT = cross(shadingS, Vector(normal: shadingNormal))
                if lengthSquared(shadingT) > 0 {
                        shadingT.normalize()
                        shadingS = cross(shadingT, Vector(normal: shadingNormal))
                } else {
                        (shadingS, shadingT) = makeCoordinateSystem(from: Vector(normal: shadingNormal))
                }

                // Set shading geometry
                dpdu = shadingS

                var faceIndex: Int = 0
                if getTriangleMeshes(scene: scene).hasFaceIndices(meshIndex: meshIndex) {
                        faceIndex = getTriangleMeshes(scene: scene).getFaceIndex(
                                meshIndex: meshIndex,
                                index: triangleIndex / 3)
                }

                // --- Finalize SurfaceInteraction ---
                interaction.position = pHitValue
                interaction.normal = normalized(normal)
                interaction.shadingNormal = normalized(shadingNormal)
                interaction.outgoing = normalized(-worldRay.direction)
                interaction.dpdu = dpdu
                interaction.uvCoordinates = uvHit
                interaction.faceIndex = faceIndex
                return interaction
        }

        func intersect(
                scene: Scene,
                ray worldRay: Ray,
                tHit: inout Real
        ) -> SurfaceInteraction? {
                var data = TriangleIntersection()
                if !getIntersectionData(scene: scene, ray: worldRay, tHit: &tHit, data: &data) {
                        return nil
                }

                // 2. Compute the full SurfaceInteraction using the new private method
                let interaction = computeSurfaceInteraction(
                        scene: scene,
                        data: data,
                        worldRay: worldRay
                )
                return interaction
        }

        private func getLocalPoint(scene: Scene, index: Int) -> Point {
                return getTriangleMeshes(scene: scene).getPointFor(meshIndex: meshIndex, at: index)
        }

        private func getWorldPoint(scene: Scene, index: Int) -> Point {
                return getLocalPoint(scene: scene, index: index)
        }

        public func getLocalPoints(scene: Scene) -> (Point, Point, Point) {
                let point0 = getLocalPoint(scene: scene, index: getVertexIndex0(scene: scene))
                let point1 = getLocalPoint(scene: scene, index: getVertexIndex1(scene: scene))
                let point2 = getLocalPoint(scene: scene, index: getVertexIndex2(scene: scene))
                return (point0, point1, point2)
        }

        public func getWorldPoints(scene: Scene) -> (Point, Point, Point) {
                let point0 = getWorldPoint(scene: scene, index: getVertexIndex0(scene: scene))
                let point1 = getWorldPoint(scene: scene, index: getVertexIndex1(scene: scene))
                let point2 = getWorldPoint(scene: scene, index: getVertexIndex2(scene: scene))
                return (point0, point1, point2)
        }

        private func uniformSampleTriangle(samples: TwoRandomVariables) -> Point2f {
                let su0 = samples.0.squareRoot()
                return Point2f(x: 1 - su0, y: samples.1 * su0)
        }

        func area(scene: Scene) -> Area {
                let (point0, point1, point2) = getWorldPoints(scene: scene)
                return 0.5 * length(cross(Vector(vector: (point1 - point0)), point2 - point0))
        }

        func sample(samples: TwoRandomVariables, scene: Scene) -> (
                interaction: SurfaceInteraction, pdf: Real
        ) {
                let barycentric = uniformSampleTriangle(samples: samples)
                let (point0, point1, point2) = getLocalPoints(scene: scene)
                let sampled0: Point = barycentric[0] * point0
                let sampled1: Point = barycentric[1] * point1
                let sampled2: Point = (1 - barycentric[0] - barycentric[1]) * point2
                let localPoint: Point = sampled0 + sampled1 + sampled2
                let localNormal = normalized(Normal(cross(point1 - point0, point2 - point0)))
                let worldPoint = getObjectToWorld(scene: scene) * localPoint
                let worldNormal = getObjectToWorld(scene: scene) * localNormal
                let worldInteraction = SurfaceInteraction(position: worldPoint, normal: worldNormal)
                let pdf = 1 / area(scene: scene)
                return (worldInteraction, pdf)
        }

        var description: String {
                var descriptionString = "Triangle [ "
                // let (p0, p1, p2) = getLocalPoints()
                // descriptionString += p0.description + p1.description + p2.description
                descriptionString += " ]"
                return descriptionString
        }

        func getObjectToWorld(scene: Scene) -> Transform {
                return getTriangleMeshes(scene: scene).getObjectToWorldFor(meshIndex: meshIndex)
        }
}

extension Triangle {
        static func createFromParameters(
                objectToWorld: Transform,
                parameters: ParameterDictionary,
                triangleMeshBuilder: TriangleMeshBuilder
        ) throws -> [ShapeType] {
                let indices = try parameters.findInts(name: "indices")
                guard indices.count % 3 == 0 else {
                        throw SceneDescriptionError.input(message: "Triangle indices must be multiplies of 3")
                }
                let points = try parameters.findPoints(name: "P")
                let normals = try parameters.findNormals(name: "N")
                let uvReals = try parameters.findReals(called: "uv")
                var uvs = [Vector2F]()
                for index in 0..<uvReals.count / 2 {
                        uvs.append(Vector2F(x: uvReals[2 * index], y: uvReals[2 * index + 1]))
                }
                let faceIndices = try parameters.findInts(name: "faceIndices")

                let meshData = MeshData(
                        indices: indices,
                        points: points,
                        normals: normals,
                        uvs: uvs,
                        faceIndices: faceIndices)

                return try createMesh(
                        objectToWorld: objectToWorld,
                        meshData: meshData,
                        triangleMeshBuilder: triangleMeshBuilder)
        }

}

struct MeshData {
        let indices: [Int]
        let points: [Point]
        let normals: [Normal]
        let uvs: [Vector2F]
        let faceIndices: [Int]
}

extension Triangle {
        static func createMesh(
                objectToWorld: Transform,
                meshData: MeshData,
                triangleMeshBuilder: TriangleMeshBuilder
        ) throws -> [ShapeType] {
                let numberTriangles = meshData.indices.count / 3
                let trianglePoints = meshData.points
                let triangleNormals = meshData.normals
                let triangleUvs = meshData.uvs
                var triangles = [ShapeType]()

                let mesh = TriangleMesh(
                        objectToWorld: objectToWorld,
                        numberTriangles: numberTriangles,
                        vertexIndices: meshData.indices,
                        points: trianglePoints,
                        normals: triangleNormals,
                        uvs: triangleUvs,
                        faceIndices: meshData.faceIndices)

                let meshIndex = triangleMeshBuilder.appendMesh(mesh: mesh)
                let triangleMeshes = triangleMeshBuilder.getMeshes()

                let pointCount = triangleMeshes.getPointCountFor(meshIndex: meshIndex)
                for triangleIndexLoop in 0..<numberTriangles {
                        let base = 3 * triangleIndexLoop
                        guard
                                triangleMeshes.getVertexIndexFor(meshIndex: meshIndex, at: base + 0)
                                        < pointCount
                                        && triangleMeshes.getVertexIndexFor(
                                                meshIndex: meshIndex, at: base + 1)
                                                < pointCount
                                        && triangleMeshes.getVertexIndexFor(
                                                meshIndex: meshIndex, at: base + 2)
                                                < pointCount
                        else {
                                throw TriangleError.index
                        }
                        triangles.append(
                                .triangle(
                                        Triangle(
                                                meshIndex: meshIndex,
                                                number: triangleIndexLoop)))
                }
                return triangles
        }
}
