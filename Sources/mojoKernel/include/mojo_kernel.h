#include <stdbool.h>
#include <stdint.h>

struct PrimId_C {
        int64_t id1;
        int64_t id2;
        int64_t materialIndex;
        int8_t type;
};

struct Material_C {
        int8_t type; // 0=none, 1=diffuse, 2=arealight
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

struct SceneDescriptor2_C {
        const struct BVH2Node *bvh2Nodes;
        const struct PrimId_C *primIds;
        const struct TriangleMesh_C *meshes;
        int64_t meshCount;
};

// CPU traversal (single ray)
void mojo_traverse_bvh2(const struct SceneDescriptor2_C *scene, const struct Ray_C *ray, float tMax,
                        struct Intersection_C *result);

// GPU support
bool mojo_gpu_available(void);

// Opaque handle to GPU-resident scene data
struct GpuSceneHandle;

void *mojo_gpu_upload_scene(
        const struct BVH2Node *bvh2Nodes,
        int64_t bvh2NodesCount,
        const struct PrimId_C *primIds,
        int64_t primIdsCount,
        const struct TriangleMesh_C *meshes,
        int64_t meshCount,
        const int64_t *meshPointsCounts,
        const int64_t *meshFaceIndicesCounts,
        const int64_t *meshVertexIndicesCounts,
        const struct Material_C *materials,
        int64_t materialCount);

void mojo_gpu_traverse_batch(
        void *handle,
        const struct Ray_C *rays,
        const float *tMaxValues,
        int64_t count,
        struct Intersection_C *results);

void mojo_gpu_shade_batch(
        void *handle,
        struct PathState_C *paths,
        int64_t count,
        const struct Intersection_C *intersections);

// CPU batch operations (parallelize over cores)
void mojo_cpu_traverse_batch(
        const struct SceneDescriptor2_C *scene,
        const struct Ray_C *rays,
        const float *tMaxValues,
        int64_t count,
        struct Intersection_C *results);

void mojo_cpu_shade_batch(
        struct PathState_C *paths,
        int64_t count,
        const struct Intersection_C *intersections,
        const struct TriangleMesh_C *meshes,
        const struct Material_C *materials);

void mojo_gpu_free_scene(void *handle);
