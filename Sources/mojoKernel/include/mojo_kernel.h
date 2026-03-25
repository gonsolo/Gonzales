#include <stdint.h>
#include <stdbool.h>

struct PrimId_C {
    int64_t id1;
    int64_t id2;
    int8_t type; 
};

struct TriangleMesh_C {
    const float* points;
    const int64_t* faceIndices;
    const int64_t* vertexIndices;
};

struct BoundingHierarchyNode; // Opaque forward declaration since Swift provides raw void pointer

struct SceneDescriptor_C {
    const void* bvhNodes;    // BoundingHierarchyNode
    const struct PrimId_C* primIds;
    const struct TriangleMesh_C* meshes;
    int64_t meshCount;
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


int32_t mojo_test_intersect(const void* bvhNodes, const struct Ray_C* ray, float tMax);
void mojo_traverse(const struct SceneDescriptor_C* scene, const struct Ray_C* ray, float tMax, struct Intersection_C* result);
