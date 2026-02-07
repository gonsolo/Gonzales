import Foundation  // sqrt, log2

@MainActor
var numberOfCurves = 0

enum CurveError: Error {
        case index
        case numberControlPoints
        case todo
        case accelerator
}

typealias TwoPoints = (Point, Point)
typealias ThreePoints = (Point, Point, Point)
// swiftlint:disable:next large_tuple
typealias FourPoints = (Point, Point, Point, Point)
// swiftlint:disable:next large_tuple
typealias SevenPoints = (Point, Point, Point, Point, Point, Point, Point)

typealias TwoFloats = (FloatX, FloatX)
typealias ThreeFloats = (FloatX, FloatX, FloatX)

private func lerp(with factor: FloatX, between first: Point, and second: Point) -> Point {
        return (1.0 - factor) * first + factor * second
}

func blossomBezier(points: FourPoints, uSamples: ThreeFloats) -> Point {
        var aPoints: ThreePoints
        aPoints.0 = lerp(with: uSamples.0, between: points.0, and: points.1)
        aPoints.1 = lerp(with: uSamples.0, between: points.1, and: points.2)
        aPoints.2 = lerp(with: uSamples.0, between: points.2, and: points.3)
        var bPoints: TwoPoints
        bPoints.0 = lerp(with: uSamples.1, between: aPoints.0, and: aPoints.1)
        bPoints.1 = lerp(with: uSamples.1, between: aPoints.1, and: aPoints.2)
        return lerp(with: uSamples.2, between: bPoints.0, and: bPoints.1)
}

func blossomBezier(points: FourPoints, uSamples: TwoFloats) -> FourPoints {
        var pPoints: FourPoints
        pPoints.0 = blossomBezier(points: points, uSamples: (uSamples.0, uSamples.0, uSamples.0))
        pPoints.1 = blossomBezier(points: points, uSamples: (uSamples.0, uSamples.0, uSamples.1))
        pPoints.2 = blossomBezier(points: points, uSamples: (uSamples.0, uSamples.1, uSamples.1))
        pPoints.3 = blossomBezier(points: points, uSamples: (uSamples.1, uSamples.1, uSamples.1))
        return pPoints
}

extension Point2 where T == FloatX {
        init(_ point: Point3) {
                self.x = point.x
                self.y = point.y
        }
}

extension Vector2 where T == FloatX {
        init(_ point: Point3) {
                self.x = point.x
                self.y = point.y
        }
}

private struct CurveIntersectionState {
        let points: [Point]
        let uRange: TwoFloats
        let depth: Int
        let index: Int
}

struct Curve: Shape {

        @MainActor
        init(objectToWorld: Transform, common: CurveCommon, uRange: TwoFloats) {
                self.common = common
                self.uRange = uRange
                self.objectToWorld = objectToWorld
                numberOfCurves += 1
        }

        func worldBound(scene: Scene) -> Bounds3f {
                return objectToWorld * objectBound(scene: scene)
        }

        func objectBound(scene _: Scene) -> Bounds3f {
                let points = blossomBezier(points: common.points, uSamples: uRange)
                let bounds = union(
                        first: Bounds3f(first: points.0, second: points.1),
                        second: Bounds3f(first: points.2, second: points.3))
                let maxWidth = computeMaxWidth(uRange: uRange, width: common.width)
                let expanded = expand(bounds: bounds, by: maxWidth * 0.5)
                return expanded
        }

        private func subdivideBezier(points: [Point]) -> [Point] {
                var splitPoints = [Point]()
                splitPoints.append(points[0])
                splitPoints.append((points[0] + points[1]) / 2)
                splitPoints.append((points[0] + 2 * points[1] + points[2]) / 4)
                splitPoints.append(
                        (points[0] + Point(3 * points[1]) + Point(3 * points[2]) + points[3]) / 8)
                splitPoints.append((points[1] + 2 * points[2] + points[3]) / 4)
                splitPoints.append((points[2] + points[3]) / 2)
                splitPoints.append(points[3])
                return splitPoints
        }

        func evalBezier(points: [Point], uSample: FloatX) -> (Point, Vector) {
                let pPoints = (points[0], points[1], points[2], points[3])
                return evalBezier(points: pPoints, uSample: uSample)
        }

        func evalBezier(points: FourPoints, uSample: FloatX) -> (Point, Vector) {
                let cp1 = ThreePoints(
                        lerp(with: uSample, between: points.0, and: points.1),
                        lerp(with: uSample, between: points.1, and: points.2),
                        lerp(with: uSample, between: points.2, and: points.3))
                let cp2 = TwoPoints(
                        lerp(with: uSample, between: cp1.0, and: cp1.1),
                        lerp(with: uSample, between: cp1.1, and: cp1.2))
                var derivative: Vector
                let directionVector: Vector = cp2.1 - cp2.0
                if lengthSquared(directionVector) > 0 {
                        derivative = 3 * directionVector
                } else {
                        derivative = points.3 - points.0
                }
                let point = lerp(with: uSample, between: cp2.0, and: cp2.1)
                return (point, derivative)
        }

        // swiftlint:disable:next function_body_length
        private func recursiveIntersect(
                ray: Ray,
                tHit: inout FloatX,
                rayToObject: Transform,
                state: CurveIntersectionState
        ) throws -> (SurfaceInteraction, FloatX) {
                let points = state.points
                let uRange = state.uRange
                let depth = state.depth
                // let index = state.index

                let nothing: (SurfaceInteraction, FloatX) = (SurfaceInteraction(), FloatX.infinity)

                let rayLength = length(ray.direction)
                if depth > 0 {
                        let splitPoints = subdivideBezier(points: points)
                        let uSamples = [uRange.0, (uRange.0 + uRange.1) / 2, uRange.1]
                        typealias InteractionAndT = (SurfaceInteraction, FloatX)
                        var hits: [InteractionAndT] = [
                                (SurfaceInteraction(), 0),
                                (SurfaceInteraction(), 0),
                        ]

                        for segment in 0..<2 {
                                let cps = segment * 3
                                let maxWidth = computeMaxWidth(
                                        uRange: (uSamples[segment], uSamples[segment + 1]), width: common.width)
                                if !overlap(
                                        points: splitPoints, index: cps, xyz: 1, width: maxWidth)
                                {
                                        continue
                                }
                                if !overlap(
                                        points: splitPoints, index: cps, xyz: 0, width: maxWidth)
                                {
                                        continue
                                }
                                let zMax = rayLength * tHit
                                if !overlap(
                                        points: splitPoints, index: cps, xyz: 2, width: maxWidth,
                                        limits: (0, zMax))
                                {
                                        continue
                                }
                                let nextState = CurveIntersectionState(
                                        points: splitPoints,
                                        uRange: (uSamples[segment], uSamples[segment + 1]),
                                        depth: depth - 1,
                                        index: cps)
                                hits[segment] = try recursiveIntersect(
                                        ray: ray,
                                        tHit: &tHit,
                                        rayToObject: rayToObject,
                                        state: nextState)
                                // if hit && !tHit: shadowRays not applicable here
                        }
                        if hits[0].0.valid && hits[1].0.valid {
                                if hits[0].1 < hits[1].1 {
                                        return hits[0]
                                } else {
                                        return hits[1]
                                }
                        } else if hits[0].0.valid && !hits[1].0.valid {
                                return hits[0]
                        } else if !hits[0].0.valid && hits[1].0.valid {
                                return hits[1]
                        } else {
                                return nothing
                        }
                } else {
                        func testTangent(_ indexA: Int, _ indexB: Int) -> Bool {
                                let edge =
                                        (points[indexA].y - points[indexB].y) * -points[indexB].y + points[indexB].x
                                        * (points[indexB].x - points[indexA].x)
                                if edge < 0 {
                                        return false
                                } else {
                                        return true
                                }
                        }

                        if !testTangent(1, 0) { return nothing }
                        if !testTangent(2, 3) { return nothing }
                        let segmentDirection: Vector2 = Point2f(points[3]) - Point2f(points[0])
                        let denominator = lengthSquared(segmentDirection)
                        if denominator == 0 { return nothing }
                        let wCoord = dot(-Vector2F(points[0]), segmentDirection) / denominator
                        let curveU = clamp(
                                value: lerp(with: wCoord, between: uRange.0, and: uRange.1),
                                low: uRange.0, high: uRange.1)
                        let hitWidth = lerp(with: curveU, between: common.width.0, and: common.width.1)
                        let (pointOnCurve, dpcdw) = evalBezier(
                                points: points, uSample: clamp(value: wCoord, low: 0, high: 1))
                        let ptCurveDist2 = pointOnCurve.x * pointOnCurve.x + pointOnCurve.y * pointOnCurve.y
                        if ptCurveDist2 > hitWidth * hitWidth * 0.25 { return nothing }
                        let zMax = rayLength * tHit
                        if pointOnCurve.z < 0 || pointOnCurve.z > zMax { return nothing }
                        let ptCurveDist = sqrt(ptCurveDist2)
                        let edgeFunc = dpcdw.x * -pointOnCurve.y + pointOnCurve.x * dpcdw.y
                        var vCoord: FloatX
                        if edgeFunc > 0 {
                                vCoord = 0.5 + ptCurveDist / hitWidth
                        } else {
                                vCoord = 0.5 - ptCurveDist / hitWidth
                        }
                        // if tHit != nullptr
                        let tHitValue = pointOnCurve.z / rayLength
                        // let pError = Vector(x: 2 * hitWidth, y: 2 * hitWidth, z: 2 * hitWidth)
                        let (_, dpdu) = evalBezier(points: common.points, uSample: curveU)
                        let dpduPlane = rayToObject.inverse * dpdu
                        let dpdvPlane =
                                normalized(Vector(x: -dpduPlane.y, y: dpduPlane.x, z: 0)) * hitWidth
                        let dpdv = rayToObject * dpdvPlane
                        let normal = Normal(normalized(cross(dpdu, dpdv)))
                        let uvHit = Point2f(x: curveU, y: vCoord)
                        let pHit = ray.getPointFor(parameter: tHitValue)
                        let localInteraction = SurfaceInteraction(
                                position: pHit,
                                normal: normal,
                                shadingNormal: normal,
                                outgoing: -ray.direction,
                                dpdu: dpdu,
                                uvCoordinates: uvHit,
                                faceIndex: 0)
                        let worldInteraction = objectToWorld * localInteraction
                        let validWorldInteraction = SurfaceInteraction(
                                valid: true,
                                position: worldInteraction.position,
                                normal: worldInteraction.normal,
                                shadingNormal: worldInteraction.shadingNormal,
                                outgoing: worldInteraction.outgoing,
                                dpdu: worldInteraction.dpdu,
                                uvCoordinates: worldInteraction.uvCoordinates,
                                faceIndex: worldInteraction.faceIndex,
                                materialIndex: worldInteraction.materialIndex)
                        return (validWorldInteraction, tHit)
                }
        }

        private func computeMaxWidth(uRange: TwoFloats, width: TwoFloats) -> FloatX {
                let width = computeWidth(uRange: uRange, width: width)
                let result = max(width.0, width.1)
                return result
        }

        private func computeWidth(uRange: TwoFloats, width: TwoFloats) -> (FloatX, FloatX) {
                let result = (
                        lerp(with: uRange.0, between: width.0, and: width.1),
                        lerp(with: uRange.1, between: width.0, and: width.1)
                )
                return result
        }

        private func overlap(
                points: [Point], index: Int = 0, xyz: Int, width: FloatX, limits: TwoFloats = (0, 0)
        ) -> Bool {
                let pMax = max(
                        max(points[index + 0][xyz], points[index + 1][xyz]),
                        max(points[index + 2][xyz], points[index + 3][xyz]))
                let pMin = min(
                        min(points[index + 0][xyz], points[index + 1][xyz]),
                        min(points[index + 2][xyz], points[index + 3][xyz]))
                var overlaps: Bool
                if pMax + 0.5 * width < limits.0 || pMin - 0.5 * width > limits.1 {
                        overlaps = false
                } else {
                        overlaps = true
                }
                return overlaps
        }

        func intersect(
                scene _: Scene,
                ray _: Ray,
                tHit _: inout FloatX
        ) throws -> Bool {
                unimplemented()
        }

        func intersect(
                scene _: Scene,
                ray _: Ray,
                tHit _: inout FloatX,
                interaction _: inout SurfaceInteraction
        ) throws {
                unimplemented()
                // let ray = worldToObject * worldRay
                // let points = blossomBezier(points: common.points, u: u)
                // let dx = cross(ray.direction, points.3 - points.0)
                // if lengthSquared(dx) == 0 {
                //        throw CurveError.todo
                // }
                // let objectToRay = try lookAtTransform(
                //        eye: ray.origin, at: ray.origin + ray.direction, up: dx)
                // var controlPoints = [Point(), Point(), Point(), Point()]
                // controlPoints[0] = objectToRay * points.0
                // controlPoints[1] = objectToRay * points.1
                // controlPoints[2] = objectToRay * points.2
                // controlPoints[3] = objectToRay * points.3
                // let maxWidth = computeMaxWidth(u: u, width: common.width)
                // if !overlap(points: controlPoints, xyz: 1, width: maxWidth) {
                //        return
                // }
                // if !overlap(points: controlPoints, xyz: 0, width: maxWidth) {
                //        return
                // }
                // let rayLength = length(ray.direction)
                // let zMax = rayLength * tHit
                // if !overlap(points: controlPoints, xyz: 2, width: maxWidth, limits: (0, zMax)) {
                //        return
                // }
                // var l0: FloatX = 0
                // for i in 0..<2 {
                //        let px: FloatX = abs(
                //                controlPoints[i].x - 2.0 * controlPoints[i + 1].x
                //                        + controlPoints[i + 2].x)
                //        let py: FloatX = abs(
                //                controlPoints[i].y - 2.0 * controlPoints[i + 1].y
                //                        + controlPoints[i + 2].y)
                //        let pz: FloatX = abs(
                //                controlPoints[i].z - 2.0 * controlPoints[i + 1].z
                //                        + controlPoints[i + 2].z)
                //        l0 = max(max(l0, px), max(py, pz))
                // }
                // let eps = max(common.width.0, common.width.1) * 0.05
                // let r0 = Int(log2(1.41421356237 * 6.0 * l0 / (8.0 * eps)) / 2.0)
                // let maxDepth = clamp(value: r0, low: 0, high: 10)
                // let interactionAndT = try recursiveIntersect(
                //        ray: ray,
                //        tHit: &tHit,
                //        points: controlPoints,
                //        rayToObject: objectToRay.inverse,
                //        u: u,
                //        depth: maxDepth)
                // tHit = interactionAndT.1
                // interaction = interactionAndT.0
        }

        func area(scene _: Scene) -> FloatX {
                fatalError("Not implemented")
        }

        func sample<I: Interaction>(samples _: TwoFloats, scene _: Scene) -> (interaction: I, pdf: FloatX) {
                fatalError("Not implemented")
        }

        public var description: String {
                return "Curve"
        }

        @MainActor
        static func statistics() {
                print("  Number of curves:\t\t\t\t\t\t\t\(numberOfCurves)")
        }

        func getObjectToWorld(scene _: Scene) -> Transform {
                return objectToWorld
        }

        let common: CurveCommon
        let uRange: TwoFloats
        let objectToWorld: Transform
}

struct CurveCommon {

        let points: FourPoints
        let width: TwoFloats
}

@MainActor
func createCurve(objectToWorld: Transform, points: FourPoints, width: TwoFloats) -> [ShapeType] {
        let common = CurveCommon(points: points, width: width)
        // The default splitdepth in pbrt is 3 which would make this 8 segments (1 << splitDepth).
        // Let's use 4 for the time being to save a little bit of memory.
        let numberOfSegments = 4
        var segments = [ShapeType]()
        for index in 0..<numberOfSegments {
                let numSegments = FloatX(numberOfSegments)
                let uMin = FloatX(index) / numSegments
                let uMax = (FloatX(index) + 1) / numSegments
                let curve = Curve(objectToWorld: objectToWorld, common: common, uRange: (uMin, uMax))
                let shape = ShapeType.curve(curve)
                segments.append(shape)
        }
        return segments
}

@MainActor
func createBVHCurveShape(
        controlPoints: [Point],
        widths: (Float, Float),
        objectToWorld: Transform,
        degree: Int
) -> [ShapeType] {
        var curves = [ShapeType]()
        let numberOfSegments = controlPoints.count - degree
        for segment in 0..<numberOfSegments {
                var bezierPoints: FourPoints
                let p012 = controlPoints[segment + 0]
                let p123 = controlPoints[segment + 1]
                let p234 = controlPoints[segment + 2]
                let p345 = controlPoints[segment + 3]
                let p122 = lerp(with: 2.0 / 3.0, between: p012, and: p123)
                let p223 = lerp(with: 1.0 / 3.0, between: p123, and: p234)
                let p233 = lerp(with: 2.0 / 3.0, between: p123, and: p234)
                let p334 = lerp(with: 1.0 / 3.0, between: p234, and: p345)
                let p222 = lerp(with: 0.5, between: p122, and: p223)
                let p333 = lerp(with: 0.5, between: p233, and: p334)
                bezierPoints.0 = p222
                bezierPoints.1 = p223
                bezierPoints.2 = p233
                bezierPoints.3 = p333
                let width0 = lerp(
                        with: FloatX(segment) / FloatX(numberOfSegments), between: widths.0,
                        and: widths.1)
                let width1 = lerp(
                        with: FloatX(segment + 1) / FloatX(numberOfSegments), between: widths.0,
                        and: widths.1)
                let curve = createCurve(
                        objectToWorld: objectToWorld,
                        points: bezierPoints,
                        width: (width0, width1))

                curves.append(contentsOf: curve)
        }
        return curves
}

@MainActor
func createCurveShape(objectToWorld: Transform, parameters: ParameterDictionary) throws -> [ShapeType] {
        let degree = 3
        let controlPoints = try parameters.findPoints(name: "P")
        let width = try parameters.findOneFloatX(called: "width", else: 0.5)
        let width0 = try parameters.findOneFloatX(called: "width0", else: width)
        let width1 = try parameters.findOneFloatX(called: "width1", else: width)
        guard controlPoints.count >= degree + 1 else {
                throw CurveError.numberControlPoints
        }
        var curves = [ShapeType]()
        switch acceleratorName {
        case "bvh":
                curves = createBVHCurveShape(
                        controlPoints: controlPoints,
                        widths: (width0, width1),
                        objectToWorld: objectToWorld,
                        degree: degree)
        case "embree":
                curves = createEmbreeCurveShape(
                        controlPoints: controlPoints,
                        widths: (width0, width1),
                        objectToWorld: objectToWorld
                )
        default:
                throw CurveError.accelerator
        }
        return curves
}
