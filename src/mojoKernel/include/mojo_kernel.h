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
        int32_t n_tris;        /* number of triangles in this light mesh */
        float emissionR, emissionG, emissionB;
        float total_area;      /* total surface area of this light mesh */
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

// Matrix math (step 13a). 4x4 matrices are 16 floats in column-major order
// (col0[0..3], col1[0..3], col2[0..3], col3[0..3]), matching
// Transform.columnMajorFloats(). out may not alias a or b.
void mojo_matrix_multiply(const float *a, const float *b, float *out);
// Invert a 4x4 matrix (Gauss-Jordan, full pivoting). Returns 1 on success;
// on a singular matrix writes the identity and returns 0.
int32_t mojo_matrix_invert(const float *m, float *out);

// Bulk geometry transform (step 13b).
// Points are 4 floats each (SIMD4 layout: x,y,z,w=1); normals are 3 floats each.
// matrix / inv_matrix are column-major (flat[col*4+row] = m[row,col]).
void mojo_transform_points(const float *matrix, const float *points_in,
                           int32_t count, float *points_out);
void mojo_transform_normals(const float *inv_matrix, const float *normals_in,
                            int32_t count, float *normals_out);

// Gaussian filter normalization (step 14).
// Returns 0.5*(1+erf(support/(sigma*sqrt(2)))), matching GaussianFilter.mojoParams().
float mojo_gaussian_norm(float support, float sigma);

// Tile scheduling (step 15). Renders every tile of the image sequentially in Mojo
// (parallelism will be added when Mojo owns main()). Results buffer must hold
// (max_x-min_x)*(max_y-min_y) TileResult_C entries, one per pixel in [py][px] order.
void mojo_render_all_tiles(
    const float *raster_to_camera, const float *camera_to_world,
    int32_t min_x, int32_t min_y, int32_t max_x, int32_t max_y,
    int32_t tile_w, int32_t tile_h,
    const struct TileSamplerParams_C *sampler_params,
    const struct SceneDescriptor2_C *scene,
    struct TileResult_C *results,
    int32_t max_depth
);

// Film normalization (step 16). Converts TileResult_C[] → per-pixel float RGB arrays.
// beauty_out and albedo_out must each hold count*3 floats (R,G,B interleaved).
// Applies iso/100 scaling and optional maxComponentValue clamping to beauty.
// Image write via OpenImageIO bridge (step 18).
// pixels: width*height*3 floats (R,G,B interleaved). filename: null-terminated UTF-8.
// Returns 1 on success, 0 on failure.
int32_t mojo_write_exr(
    const float *pixels, int32_t width, int32_t height,
    const uint8_t *filename, int32_t tile_w, int32_t tile_h
);

void mojo_normalize_film(
    const struct TileResult_C *results, int32_t count,
    float iso, float max_component_value,
    float *beauty_out, float *albedo_out
);

// Joint bilateral denoiser guided by albedo (step 17).
// beauty and albedo are width*height*3 float arrays (R,G,B interleaved, row-major).
// output must hold width*height*3 floats.
// radius: filter half-width; sigma_s: spatial sigma; sigma_r: albedo-range sigma.
void mojo_denoise(
    const float *beauty, const float *albedo,
    int32_t width, int32_t height,
    float *output,
    int32_t radius, float sigma_s, float sigma_r
);

// PbrtScanner handle-based API (step 19).
// Opaque handle type — allocates buffer + state on heap.
typedef struct PbrtScanner_Mojo PbrtScanner_Mojo;

PbrtScanner_Mojo *mojo_scanner_new(const uint8_t *path);
PbrtScanner_Mojo *mojo_scanner_new_from_bytes(const uint8_t *bytes, int32_t length);
void               mojo_scanner_free(PbrtScanner_Mojo *handle);

int32_t mojo_scanner_is_at_end(PbrtScanner_Mojo *handle);
int32_t mojo_scanner_scan_location(PbrtScanner_Mojo *handle);

int32_t mojo_scanner_peek_char(PbrtScanner_Mojo *handle, uint8_t expected);
int32_t mojo_scanner_scan_char(PbrtScanner_Mojo *handle, uint8_t expected);

int32_t mojo_scanner_scan_int(PbrtScanner_Mojo *handle, int32_t *result);
int32_t mojo_scanner_scan_float(PbrtScanner_Mojo *handle, float *result);

int32_t mojo_scanner_count_floats(PbrtScanner_Mojo *handle);
int32_t mojo_scanner_scan_floats(PbrtScanner_Mojo *handle, float *out, int32_t max_count);

int32_t mojo_scanner_count_ints(PbrtScanner_Mojo *handle);
int32_t mojo_scanner_scan_ints(PbrtScanner_Mojo *handle, int32_t *out, int32_t max_count);

// Returns bytes written into buf (excl. NUL), or -1 on failure.
int32_t mojo_scanner_parse_quoted_string(PbrtScanner_Mojo *handle, uint8_t *buf, int32_t max_buf);

// Returns 1 if header found, 0 otherwise. Sets *is_array if '[' follows.
int32_t mojo_scanner_parse_param_header(
    PbrtScanner_Mojo *handle,
    uint8_t *type_buf, int32_t type_max,
    uint8_t *name_buf, int32_t name_max,
    int32_t *is_array
);

// Returns bytes written, 0 for empty token, <0 at end-of-file.
int32_t mojo_scanner_scan_token(
    PbrtScanner_Mojo *handle,
    const uint8_t *delims, int32_t n_delims,
    uint8_t *buf, int32_t max_buf
);

// Full .pbrt scene parse + BVH build (step 20).
// Returns an opaque handle; caller must call mojo_parsed_free when done.
typedef struct ParsedScene_Mojo ParsedScene_Mojo;

ParsedScene_Mojo *mojo_parse_scene(const uint8_t *path);
void              mojo_parsed_free(ParsedScene_Mojo *handle);

// Build SceneDescriptor2_C from a parsed scene. Caller must free the returned pointer.
struct SceneDescriptor2_C *mojo_parsed_scene_descriptor(ParsedScene_Mojo *handle);

// Getters for render parameters from a parsed scene.
void    mojo_parsed_raster_to_camera(ParsedScene_Mojo *handle, float *out16);
void    mojo_parsed_camera_to_world (ParsedScene_Mojo *handle, float *out16);
int32_t mojo_parsed_film_w          (ParsedScene_Mojo *handle);
int32_t mojo_parsed_film_h          (ParsedScene_Mojo *handle);
float   mojo_parsed_film_iso        (ParsedScene_Mojo *handle);
float   mojo_parsed_film_max_comp   (ParsedScene_Mojo *handle);
void    mojo_parsed_film_filename   (ParsedScene_Mojo *handle, char *buf, int32_t max_buf);
float   mojo_parsed_filter_sigma    (ParsedScene_Mojo *handle);
float   mojo_parsed_filter_support_x(ParsedScene_Mojo *handle);
float   mojo_parsed_filter_support_y(ParsedScene_Mojo *handle);
float   mojo_parsed_filter_norm_x   (ParsedScene_Mojo *handle);
float   mojo_parsed_filter_norm_y   (ParsedScene_Mojo *handle);
float   mojo_parsed_filter_weight   (ParsedScene_Mojo *handle);
int32_t mojo_parsed_samples_per_pixel(ParsedScene_Mojo *handle);
int32_t mojo_parsed_log2_spp        (ParsedScene_Mojo *handle);
int32_t mojo_parsed_n_base4_digits  (ParsedScene_Mojo *handle);
int32_t mojo_parsed_max_depth       (ParsedScene_Mojo *handle);
uint64_t mojo_parsed_rng_seed       (ParsedScene_Mojo *handle);

// Parse .pbrt file and render to EXR in one call.
// sobol_matrices must point to 21201*52 uint32 values (Joe-Kuo table).
int32_t mojo_parse_and_render(const uint8_t *path, const uint32_t *sobol_matrices);
