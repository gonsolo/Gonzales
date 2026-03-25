struct BoundingHierarchyNode {

        var pMinX = SIMD8<Float>(repeating: .infinity)
        var pMaxX = SIMD8<Float>(repeating: -.infinity)
        var pMinY = SIMD8<Float>(repeating: .infinity)
        var pMaxY = SIMD8<Float>(repeating: -.infinity)
        var pMinZ = SIMD8<Float>(repeating: .infinity)
        var pMaxZ = SIMD8<Float>(repeating: -.infinity)

        var childNodes = SIMD8<Int32>(repeating: -1)
        var primitiveOffsets = SIMD8<Int32>(repeating: 0)
        var primitiveCounts = SIMD8<Int32>(repeating: 0)

        struct RayAABBPrecomputed {
                let rdirX: SIMD8<Float>, rdirY: SIMD8<Float>, rdirZ: SIMD8<Float>
                let orgRdirX: SIMD8<Float>, orgRdirY: SIMD8<Float>, orgRdirZ: SIMD8<Float>
                let nearXIsMin: Bool, nearYIsMin: Bool, nearZIsMin: Bool
        }

        /// Embree-style AABB test: precomputed orgRdir + near/far bound selection
        /// eliminates per-axis min/max and uses FMA-candidate multiply-subtract.
        @inline(__always)
        func intersect8(
                ray: RayAABBPrecomputed,
                tHit: Float
        ) -> (SIMD8<Float>, SIMDMask<SIMD8<Float.SIMDMaskScalar>>) {
                // Select near/far bounds based on ray direction sign (precomputed per ray)
                // This eliminates the per-axis min/max operations entirely
                let nearX = ray.nearXIsMin ? pMinX : pMaxX
                let farX  = ray.nearXIsMin ? pMaxX : pMinX
                let nearY = ray.nearYIsMin ? pMinY : pMaxY
                let farY  = ray.nearYIsMin ? pMaxY : pMinY
                let nearZ = ray.nearZIsMin ? pMinZ : pMaxZ
                let farZ  = ray.nearZIsMin ? pMaxZ : pMinZ

                // nearBound * rdir - orgRdir = (nearBound - org) / dir
                // The compiler should fuse this into vfmsub (FMA) with -Ounchecked + AVX2
                let tNearX = nearX * ray.rdirX - ray.orgRdirX
                let tNearY = nearY * ray.rdirY - ray.orgRdirY
                let tNearZ = nearZ * ray.rdirZ - ray.orgRdirZ

                let tFarX = farX * ray.rdirX - ray.orgRdirX
                let tFarY = farY * ray.rdirY - ray.orgRdirY
                let tFarZ = farZ * ray.rdirZ - ray.orgRdirZ

                // tNear = max(tNearX, tNearY, tNearZ, 0)
                let zero = SIMD8<Float>(repeating: 0)
                var tNear = tNearX.replacing(with: tNearY, where: tNearY .> tNearX)
                tNear = tNear.replacing(with: tNearZ, where: tNearZ .> tNear)
                tNear = tNear.replacing(with: zero, where: zero .> tNear)

                // tFar = min(tFarX, tFarY, tFarZ, tHit) * gamma
                let tHitV = SIMD8<Float>(repeating: tHit)
                var tFar = tFarX.replacing(with: tFarY, where: tFarY .< tFarX)
                tFar = tFar.replacing(with: tFarZ, where: tFarZ .< tFar)
                tFar = tFar.replacing(with: tHitV, where: tHitV .< tFar)

                let safeGamma = SIMD8<Float>(repeating: 1.0000003)
                tFar *= safeGamma

                return (tNear, tNear .<= tFar)
        }
}
