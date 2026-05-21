#include <stdbool.h>
#include <stdint.h>

struct PrimId_C {
        int64_t id1;
        int64_t id2;
        int64_t materialIndex;
        int8_t type;
};

struct Material_C {
        int8_t type; // 0=none, 1=diffuse, 2=arealight, 3=conductor, 4=dielectric, 5=coatedDiffuse, 6=diffuseTransmission
        float albedo[3];
        float emission[3];
};

struct TriangleMesh_C {
        const float *points;
        const int64_t *faceIndices;
        const int64_t *vertexIndices;
};

struct Ray_C {
        float orgX, orgY, orgZ;
        float dirX, dirY, dirZ;
};

struct Intersection_C {
        struct PrimId_C primId;
        float tHit;
        float u;
        float v;
        int8_t hit;
};

struct PathState_C {
        struct Ray_C ray;
        float throughput[3];
        float estimate[3];
        float albedo[3];
        int32_t bounce;     /* path bounce counter; 0 = camera ray, incremented per diffuse hit */
        uint64_t pcgState;
        uint64_t pcgInc;
        int8_t active;
};

struct BVH2Node {
        float boundsMinX, boundsMinY, boundsMinZ;
        float boundsMaxX, boundsMaxY, boundsMaxZ;
        int32_t offset; // interior: right child index, leaf: primIds offset
        int32_t count;  // 0 = interior, >0 = leaf primitive count
};

/* Emissive triangle for next-event estimation */
struct AreaLight_C {
        int32_t meshIdx;
        int32_t triBaseVidx;   /* base index into vertexIndices (= triNum * 3) */
        float emissionR, emissionG, emissionB;
        int32_t _pad;
};

struct SceneDescriptor2_C {
        const struct BVH2Node *bvh2Nodes;
        const struct PrimId_C *primIds;
        const struct TriangleMesh_C *meshes;
        int64_t meshCount;
        const struct Material_C *materials;
        int64_t materialCount;
        const struct AreaLight_C *areaLights;
        int64_t areaLightCount;
};

// CPU traversal (single ray)
void mojo_traverse_bvh2(const struct SceneDescriptor2_C *scene, const struct Ray_C *ray, float tMax,
                        struct Intersection_C *result);

// GPU support
bool mojo_gpu_available(void);

// Opaque handle to GPU-resident scene data
struct GpuSceneHandle;

void *mojo_gpu_upload_scene(const struct BVH2Node *bvh2Nodes, int64_t bvh2NodesCount,
                            const struct PrimId_C *primIds, int64_t primIdsCount,
                            const struct TriangleMesh_C *meshes, int64_t meshCount,
                            const int64_t *meshPointsCounts, const int64_t *meshFaceIndicesCounts,
                            const int64_t *meshVertexIndicesCounts, const struct Material_C *materials,
                            int64_t materialCount);

void mojo_gpu_traverse_batch(void *handle, const struct Ray_C *rays, const float *tMaxValues, int64_t count,
                             struct Intersection_C *results);

void mojo_gpu_shade_batch(void *handle, struct PathState_C *paths, int64_t count,
                          const struct Intersection_C *intersections);

// CPU batch operations (parallelize over cores)
void mojo_cpu_traverse_batch(const struct SceneDescriptor2_C *scene, const struct Ray_C *rays,
                             const float *tMaxValues, int64_t count, struct Intersection_C *results);

void mojo_cpu_shade_batch(struct PathState_C *paths, int64_t count,
                          const struct Intersection_C *intersections, const struct TriangleMesh_C *meshes,
                          const struct Material_C *materials);

void mojo_gpu_free_scene(void *handle);

// Full multi-bounce CPU path trace for a batch of paths (replaces the Swift bounce loop)
void mojo_render_paths(const struct SceneDescriptor2_C *scene, struct PathState_C *paths,
                       int64_t count, int32_t maxDepth);

// Per-sample data from Swift's Sobol sampler
struct PixelSample_C {
        float filmX;
        float filmY;
        float filterWeight;
        int32_t pixelX;
        int32_t pixelY;
        int32_t _pad;        /* align pcgState to 8 bytes */
        uint64_t pcgState;
        uint64_t pcgInc;
};

// Per-sample result returned to Swift for film accumulation
struct TileResult_C {
        float estimateR, estimateG, estimateB;
        float albedoR, albedoG, albedoB;
        float filterWeight;
        int32_t pixelX, pixelY;
};

// Parameters for mojo_render_tile_v2: Sobol sampler + Gaussian filter, all evaluated in Mojo.
// sobolMatrices must point to NSobolDimensions * SobolMatrixSize uint32 values (standard Joe-Kuo table).
// filterNorm{X,Y} = 0.5*(1+erf(support/(sigma*sqrt(2)))), pre-computed by Swift using Foundation.erf.
// filterWeight   = (2*filterNormX-1)*(2*filterNormY-1) — the constant importance-sampling weight.
// rngSeed is hashed with pixel coords + sample index to derive per-path PCG seeds.
struct TileSamplerParams_C {
        const uint32_t *sobolMatrices;
        uint64_t rngSeed;
        int32_t sobolSeed;
        int32_t log2SamplesPerPixel;
        int32_t nBase4Digits;
        int32_t samplesPerPixel;
        float filterSigma;
        float filterSupportX;
        float filterSupportY;
        float filterNormX;
        float filterNormY;
        float filterWeight;
};

// Build a depth-first BVH2 over `primCount` primitive AABBs (primBounds: 6 floats
// per prim — minX,minY,minZ,maxX,maxY,maxZ). outNodes must hold >= 2*primCount nodes,
// outOrder >= primCount int32s. Writes the node array and the primitive permutation
// (outOrder[k] = original index now at leaf position k); returns the node count.
int32_t mojo_build_bvh2(
        const float *primBounds, int32_t primCount,
        struct BVH2Node *outNodes, int32_t *outOrder
);

// Camera ray generation + full path trace + Sobol film sampling — all in Mojo.
// Samples are accumulated per pixel inside the kernel, so results must be pre-allocated
// to (tileMaxX-tileMinX)*(tileMaxY-tileMinY) entries (one per pixel), filled in [iy][ix]
// order with pixel coords and the summed filterWeight written.
void mojo_render_tile_v2(
        const float *rasterToCamera, const float *cameraToWorld,
        int32_t tileMinX, int32_t tileMinY, int32_t tileMaxX, int32_t tileMaxY,
        const struct TileSamplerParams_C *samplerParams,
        const struct SceneDescriptor2_C *scene,
        struct TileResult_C *results,
        int32_t maxDepth
);

// Camera ray generation + full path trace in Mojo.
// rasterToCamera and cameraToWorld are 16-element column-major float arrays.
void mojo_render_tile(const float *rasterToCamera, const float *cameraToWorld,
                      const struct PixelSample_C *samples, int64_t count,
                      const struct SceneDescriptor2_C *scene,
                      struct TileResult_C *results, int32_t maxDepth);

// pbrt lexer — numeric scanning over a whole-file byte buffer (step 11b).
// cursor is read and updated in place. Returns 1 on success, 0 if no number found.
int32_t mojo_scan_int(const uint8_t *bytes, int32_t len,
                      int32_t *cursor, int32_t *result);
int32_t mojo_scan_float(const uint8_t *bytes, int32_t len,
                        int32_t *cursor, float *result);

// pbrt lexer — bulk array scan (step 11c).
// mojo_count_* takes cursor by value (no side effects); returns element count.
// mojo_scan_* fills out[], advances *cursor to the stop position, returns count written.
int32_t mojo_count_floats(const uint8_t *bytes, int32_t len, int32_t cursor);
int32_t mojo_scan_floats(const uint8_t *bytes, int32_t len, int32_t *cursor,
                         float *out, int32_t max_count);
int32_t mojo_count_ints(const uint8_t *bytes, int32_t len, int32_t cursor);
int32_t mojo_scan_ints(const uint8_t *bytes, int32_t len, int32_t *cursor,
                       int32_t *out, int32_t max_count);

// pbrt lexer — token/keyword scanning (step 11d).
// mojo_scan_char: skip ws, consume expected byte; returns 1 on match, 0 otherwise.
//   *cursor is always advanced past leading whitespace.
// mojo_peek_char: same but does NOT consume the matched byte.
// mojo_scan_token: skip ws, read bytes into buf until a delimiter or EOF.
//   Returns bytes written (>= 0) or -1 if EOF before any content.
//   buf is null-terminated; at most max_buf-1 bytes are written.
int32_t mojo_scan_char(const uint8_t *bytes, int32_t len, int32_t *cursor,
                       uint8_t expected);
int32_t mojo_peek_char(const uint8_t *bytes, int32_t len, int32_t *cursor,
                       uint8_t expected);
int32_t mojo_scan_token(const uint8_t *bytes, int32_t len, int32_t *cursor,
                        const uint8_t *delims, int32_t n_delims,
                        uint8_t *buf, int32_t max_buf);

// pbrt parser — directive-level (step 12).
// mojo_parse_quoted_string: skip ws, read "content", write content to buf.
//   Returns bytes written >= 0, or -1 if no opening '"'.
// mojo_parse_param_header: read "type name" and optional '['.
//   Writes null-terminated type and name; sets *is_array (1 if '[' consumed).
//   Returns 1 if found, 0 if no opening '"'.
int32_t mojo_parse_quoted_string(const uint8_t *bytes, int32_t len, int32_t *cursor,
                                 uint8_t *buf, int32_t max_buf);
int32_t mojo_parse_param_header(const uint8_t *bytes, int32_t len, int32_t *cursor,
                                uint8_t *type_buf, int32_t type_max,
                                uint8_t *name_buf, int32_t name_max,
                                int32_t *is_array);
