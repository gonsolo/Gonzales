// Catmull-Clark meshes as bicubic Bezier patches for direct ray tracing (Binder and Keller 2018). OpenSubdiv isolates the
// irregular vertices (feature-adaptive refinement); every patch, regular or not, is then fitted as one Bezier patch from 16
// samples of OpenSubdiv's own limit surface: exact for regular patches (boundaries and sharp corners included), the
// approximation of Tejima et al. 2015 for the Gregory patches around extraordinary vertices.
#include <opensubdiv/far/patchTableFactory.h>
#include <opensubdiv/far/primvarRefiner.h>
#include <opensubdiv/far/stencilTable.h>
#include <opensubdiv/far/topologyDescriptor.h>
#include <opensubdiv/far/topologyRefinerFactory.h>
#include <cstring>
#include <memory>
#include <vector>

using namespace OpenSubdiv;

namespace {

struct Patches {
        std::vector<float> cp;    // 16 control points (x, y, z) a patch, row v = 0 first
        std::vector<float> uv;    // the patch's square in its base face: u0, v0, size (ptex style)
        std::vector<int> face;    // its Ptex face: a quad, or one sub-face of another polygon
};

// Inverse of the cubic Bernstein matrix at t = 0, 1/3, 2/3, 1: Bezier points from 4 samples of a cubic.
const float kInv[4][4] = {{1, 0, 0, 0}, {-5.0f / 6, 3, -1.5f, 1.0f / 3}, {1.0f / 3, -1.5f, 3, -5.0f / 6}, {0, 0, 0, 1}};

}  // namespace

extern "C" {

// Patches of a Catmull-Clark mesh; creases are vertex pairs with a sharpness each. *n_out is the patch count.
void *osd_build(int n_verts, const float *pts, int n_faces, const int *counts, const int *indices,
                int n_creases, const int *crease_verts, const float *crease_sharp,
                int n_corners, const int *corner_verts, const float *corner_sharp, int isolation, int *n_out) {
        *n_out = 0;
        Far::TopologyDescriptor desc;
        desc.numVertices = n_verts;
        desc.numFaces = n_faces;
        desc.numVertsPerFace = counts;
        desc.vertIndicesPerFace = indices;
        desc.numCreases = n_creases;
        desc.creaseVertexIndexPairs = crease_verts;
        desc.creaseWeights = crease_sharp;
        desc.numCorners = n_corners;
        desc.cornerVertexIndices = corner_verts;
        desc.cornerWeights = corner_sharp;
        Sdc::Options sdc;
        sdc.SetVtxBoundaryInterpolation(Sdc::Options::VTX_BOUNDARY_EDGE_ONLY);
        using Factory = Far::TopologyRefinerFactory<Far::TopologyDescriptor>;
        std::unique_ptr<Far::TopologyRefiner> refiner(
            Factory::Create(desc, Factory::Options(Sdc::SCHEME_CATMARK, sdc)));
        if (!refiner)
                return nullptr;
        Far::PatchTableFactory::Options po(isolation);
        po.SetEndCapType(Far::PatchTableFactory::Options::ENDCAP_GREGORY_BASIS);
        po.useInfSharpPatch = true;
        refiner->RefineAdaptive(po.GetRefineAdaptiveOptions());
        std::unique_ptr<Far::PatchTable> table(Far::PatchTableFactory::Create(*refiner, po));
        if (!table)
                return nullptr;

        // Every refined vertex, then the patch table's local points (Gregory basis points).
        const int n_refined = refiner->GetNumVerticesTotal(), n_local = table->GetNumLocalPoints();
        std::vector<float> verts(static_cast<size_t>(n_refined + n_local) * 3);
        std::memcpy(verts.data(), pts, static_cast<size_t>(n_verts) * 3 * sizeof(float));
        struct V {
                float *p;
                void Clear() { p[0] = p[1] = p[2] = 0; }
                void AddWithWeight(V const &s, float w) { p[0] += w * s.p[0]; p[1] += w * s.p[1]; p[2] += w * s.p[2]; }
        };
        std::vector<V> vv(n_refined + n_local);
        for (size_t i = 0; i < vv.size(); ++i)
                vv[i].p = &verts[i * 3];
        Far::PrimvarRefiner primvar(*refiner);
        int start = 0;
        for (int level = 1; level <= refiner->GetMaxLevel(); ++level) {
                const int n_prev = refiner->GetLevel(level - 1).GetNumVertices();
                V *src = &vv[start], *dst = &vv[start + n_prev];
                primvar.Interpolate(level, src, dst);
                start += n_prev;
        }
        if (n_local > 0)
                table->ComputeLocalPointValues(&vv[0], &vv[n_refined]);

        auto *out = new Patches;
        float w[20];
        int patch_base = 0, vert_base = 0;
        for (int a = 0; a < table->GetNumPatchArrays(); ++a) {
                const int n = table->GetNumPatches(a), ncv = table->GetPatchArrayDescriptor(a).GetNumControlVertices();
                Far::ConstIndexArray cvs = table->GetPatchArrayVertices(a);
                for (int p = 0; p < n; ++p) {
                        Far::PatchTable::PatchHandle h;
                        h.arrayIndex = a;
                        h.patchIndex = patch_base + p;
                        h.vertIndex = vert_base + p * ncv;
                        const Far::PatchParam param = table->GetPatchParam(a, p);
                        float s[4][4][3];
                        for (int j = 0; j < 4; ++j)
                                for (int i = 0; i < 4; ++i) {
                                        float u = i / 3.0f, v = j / 3.0f;
                                        param.Unnormalize(u, v);
                                        table->EvaluateBasis(h, u, v, w);
                                        for (int c = 0; c < 3; ++c) {
                                                float acc = 0;
                                                for (int k = 0; k < ncv; ++k)
                                                        acc += w[k] * verts[static_cast<size_t>(cvs[p * ncv + k]) * 3 + c];
                                                s[j][i][c] = acc;
                                        }
                                }
                        float t[4][4][3], b[4][4][3];
                        for (int j = 0; j < 4; ++j)   // samples to Bezier points along u, then along v
                                for (int i = 0; i < 4; ++i)
                                        for (int c = 0; c < 3; ++c) {
                                                t[j][i][c] = 0;
                                                for (int k = 0; k < 4; ++k)
                                                        t[j][i][c] += kInv[i][k] * s[j][k][c];
                                        }
                        for (int j = 0; j < 4; ++j)
                                for (int i = 0; i < 4; ++i)
                                        for (int c = 0; c < 3; ++c) {
                                                b[j][i][c] = 0;
                                                for (int k = 0; k < 4; ++k)
                                                        b[j][i][c] += kInv[j][k] * t[k][i][c];
                                                out->cp.push_back(b[j][i][c]);
                                        }
                        float u0 = 0, v0 = 0, u1 = 1, v1 = 1;
                        param.Unnormalize(u0, v0);
                        param.Unnormalize(u1, v1);
                        out->uv.insert(out->uv.end(), {u0, v0, u1 - u0});
                        out->face.push_back(param.GetFaceId());
                }
                patch_base += n;
                vert_base += n * ncv;
        }
        *n_out = static_cast<int>(out->face.size());
        return out;
}

// Copy the patches into cp (48 floats a patch), uv (3 floats) and face (1 int), then free them.
void osd_take(void *h, float *cp, float *uv, int *face) {
        auto *p = static_cast<Patches *>(h);
        std::memcpy(cp, p->cp.data(), p->cp.size() * sizeof(float));
        std::memcpy(uv, p->uv.data(), p->uv.size() * sizeof(float));
        std::memcpy(face, p->face.data(), p->face.size() * sizeof(int));
        delete p;
}

}  // extern "C"
