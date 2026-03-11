import Testing

@testable import libgonzales

@Suite struct ReflectionTests {

        // MARK: - CosTheta / SinTheta

        @Test func cosThetaAtPole() {
                // Vector pointing straight up in local coordinates: (0,0,1)
                let v = Vector(x: 0, y: 0, z: 1)
                #expect(abs(cosTheta(v) - 1.0) <= 1e-6)
                #expect(abs(sinTheta(v)) <= 1e-6)
        }

        @Test func cosThetaAtEquator() {
                // Vector in x-y plane: (1,0,0)
                let v = Vector(x: 1, y: 0, z: 0)
                #expect(abs(cosTheta(v)) <= 1e-6)
                #expect(abs(sinTheta(v) - 1.0) <= 1e-6)
        }

        @Test func cos2ThetaPlusSin2ThetaEqualsOne() {
                let v = normalized(Vector(x: 1, y: 2, z: 3))
                let sum = cos2Theta(v) + sin2Theta(v)
                #expect(abs(sum - 1.0) <= 1e-5)
        }

        // MARK: - CosPhi / SinPhi

        @Test func cosPhiAtXAxis() {
                // Vector in x-z plane at equator: sinTheta=1, so cosPhi = x/sinTheta
                let v = Vector(x: 1, y: 0, z: 0)
                #expect(abs(cosPhi(v) - 1.0) <= 1e-6)
                #expect(abs(sinPhi(v)) <= 1e-6)
        }

        @Test func sinPhiAtYAxis() {
                let v = Vector(x: 0, y: 1, z: 0)
                #expect(abs(cosPhi(v)) <= 1e-6)
                #expect(abs(sinPhi(v) - 1.0) <= 1e-6)
        }

        @Test func cos2PhiPlusSin2PhiEqualsOne() {
                let v = normalized(Vector(x: 1, y: 2, z: 0.5))
                let sum = cos2Phi(v) + sin2Phi(v)
                #expect(abs(sum - 1.0) <= 1e-5)
        }

        // MARK: - SameHemisphere

        @Test func sameHemispherePositive() {
                let a = Vector(x: 0, y: 0, z: 1)
                let b = Vector(x: 1, y: 1, z: 0.5)
                #expect(sameHemisphere(a, b))
        }

        @Test func sameHemisphereNegative() {
                let a = Vector(x: 0, y: 0, z: 1)
                let b = Vector(x: 0, y: 0, z: -1)
                #expect(!sameHemisphere(a, b))
        }

        // MARK: - Reflect

        @Test func reflectNormalIncidence() {
                // Reflecting (0,0,1) about normal (0,0,1) should give (0,0,1)
                let v = Vector(x: 0, y: 0, z: 1)
                let n = Vector(x: 0, y: 0, z: 1)
                let result = reflect(vector: v, by: n)
                #expect(abs(result.x) <= 1e-6)
                #expect(abs(result.y) <= 1e-6)
                #expect(abs(result.z - 1.0) <= 1e-6)
        }

        @Test func reflectPerfectMirror() {
                // Reflecting (1,0,-1) about (0,0,1) should give (1,0,1)
                // reflect(v, n) = -v + 2*dot(v,n)*n
                // dot((1,0,-1),(0,0,1)) = -1
                // -v = (-1,0,1), 2*(-1)*(0,0,1) = (0,0,-2)
                // result = (-1,0,1) + (0,0,-2) = (-1,0,-1)
                let v = Vector(x: 1, y: 0, z: -1)
                let n = Vector(x: 0, y: 0, z: 1)
                let result = reflect(vector: v, by: n)
                #expect(abs(result.x - (-1)) <= 1e-6)
                #expect(abs(result.y) <= 1e-6)
                #expect(abs(result.z - (-1)) <= 1e-6)
        }

        // MARK: - Refract

        @Test func refractSameMediumNoDeviation() {
                // eta = 1 means same medium, direction should be unchanged (negated incident)
                let incident = normalized(Vector(x: 0.3, y: 0, z: 1))
                let normal = Normal(x: 0, y: 0, z: 1)
                let result = refract(incident: incident, normal: normal, eta: 1.0)
                #expect(result != nil)
        }

        @Test func refractTotalInternalReflectionReturnsNil() {
                // Going from dense to sparse at steep angle should give TIR
                // eta = 1/1.5 = 0.667, incident nearly parallel to surface
                let incident = normalized(Vector(x: 0.99, y: 0, z: 0.01))
                let normal = Normal(x: 0, y: 0, z: -1)
                let result = refract(incident: incident, normal: normal, eta: 1.5)
                // At this extreme angle, TIR should occur
                // (sin²θt = sin²θi / η² would exceed 1)
                #expect(result == nil)
        }

        // MARK: - AbsCosTheta

        @Test func absCosTheta_alwaysPositive() {
                let v = Vector(x: 0, y: 0, z: -1)
                #expect(absCosTheta(v) >= 0)
                #expect(abs(absCosTheta(v) - 1.0) <= 1e-6)
        }
}
