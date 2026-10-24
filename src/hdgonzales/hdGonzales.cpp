// Hydra render delegate: collects meshes, materials, lights and the camera,
// and renders them with libgonzales.so.
#include "gonzales.h"

#include "pxr/base/gf/half.h"
#include "pxr/base/gf/matrix3d.h"
#include "pxr/base/gf/matrix4f.h"
#include "pxr/base/tf/getenv.h"
#include "pxr/imaging/hd/basisCurves.h"
#include "pxr/imaging/hd/camera.h"
#include "pxr/imaging/hd/instancer.h"
#include "pxr/imaging/hd/light.h"
#include "pxr/imaging/hd/material.h"
#include "pxr/imaging/hd/mesh.h"
#include "pxr/imaging/hd/meshUtil.h"
#include "pxr/imaging/hd/renderBuffer.h"
#include "pxr/imaging/hd/renderDelegate.h"
#include "pxr/imaging/hd/renderIndex.h"
#include "pxr/imaging/hd/renderPass.h"
#include "pxr/imaging/hd/renderPassState.h"
#include "pxr/imaging/hd/renderThread.h"
#include "pxr/imaging/hd/rendererPlugin.h"
#include "pxr/imaging/hd/rendererPluginRegistry.h"
#include "pxr/imaging/hd/resourceRegistry.h"
#include "pxr/imaging/hd/sceneDelegate.h"
#include "pxr/imaging/hd/tokens.h"
#include "pxr/imaging/hio/image.h"
#include "pxr/usd/sdf/assetPath.h"
#include "pxr/usd/usdLux/blackbody.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <fstream>
#include <functional>
#include <map>
#include <mutex>
#include <numeric>
#include <sstream>
#include <vector>

PXR_NAMESPACE_OPEN_SCOPE

namespace {

// Flattened: one vertex per triangle corner, so face-varying primvars need no indices.
struct MeshData {
    std::vector<float> points;
    std::vector<int> indices;
    std::vector<float> normals;  // world space
    std::vector<float> uvs;
    GfMatrix4f xform;
    SdfPath material;
    GfVec3f displayColor{0.5f};
};

// Round cubic B-spline segments in world space, Gonzales's curve primitive.
struct CurvesData {
    std::vector<float> controlPoints;  // 4 points per segment
    std::vector<float> widths;         // 2 diameters per segment
    std::vector<float> lookup;         // per segment: the strand's root (u, v), intercept and random
    SdfPath material;
    GfVec3f displayColor{0.5f};
};

struct MaterialData {
    GfVec3f diffuse{0.18f};
    std::string texture, normalMap;
};

struct LightData {
    TfToken type;
    GfVec3f rgb;  // irradiance (distant), radiance (dome, sphere) or radiant intensity (point)
    GfVec3f dir;
    GfVec3f pos;
    float radius;
    std::string texture;
    GfMatrix4f xform;                 // rect and disk lights: an emitting mesh
    std::vector<float> points;
    std::vector<int> indices;
};

// pbrt-v4's EqualAreaSquareToSphere, the mapping of Gonzales's environment maps.
GfVec3d _EqualAreaSquareToSphere(double px, double py) {
    double u = 2 * px - 1, v = 2 * py - 1;
    double up = std::abs(u), vp = std::abs(v);
    double signedDistance = 1 - (up + vp), r = 1 - std::abs(signedDistance);
    double phi = (r == 0 ? 1 : (vp - up) / r + 1) * M_PI / 4;
    double z = std::copysign(1 - r * r, signedDistance);
    double s = r * std::sqrt(std::max(0.0, 2 - r * r));
    return GfVec3d(std::copysign(std::cos(phi), u) * s, std::copysign(std::sin(phi), v) * s, z);
}

// Resamples a lat-long image into an equal-area square in world space and returns the path of that image.
// `st` maps a world direction to the image's (s, t), t = 0 at the top row.
std::string _EqualArea(std::string const &path, std::string const &key,
                       std::function<std::pair<double, double>(GfVec3d const &)> const &st) {
    HioImageSharedPtr in = HioImage::OpenForReading(path);
    if (!in)
        return {};
    int w = in->GetWidth(), h = in->GetHeight();
    std::vector<GfVec3f> src(size_t(w) * h);
    HioImage::StorageSpec spec;
    spec.width = w; spec.height = h; spec.depth = 1;
    spec.format = HioFormatFloat32Vec3;
    spec.data = src.data();
    if (!in->Read(spec))
        return {};
    int n = std::clamp(int(std::sqrt(double(w) * h)), 16, 4096);
    std::vector<GfVec3f> dst(size_t(n) * n);
    for (int y = 0; y < n; ++y)
        for (int x = 0; x < n; ++x) {
            auto [s, t] = st(_EqualAreaSquareToSphere((x + 0.5) / n, (y + 0.5) / n));
            dst[size_t(y) * n + x] = src[size_t(std::clamp(int(t * h), 0, h - 1)) * w + std::clamp(int(s * w), 0, w - 1)];
        }
    std::string out = TfGetenv("TMPDIR", "/tmp") + "/gonzales_dome_" + std::to_string(std::hash<std::string>{}(path + key)) + ".exr";
    HioImageSharedPtr image = HioImage::OpenForWriting(out);
    spec.width = spec.height = n;
    spec.data = dst.data();
    return image && image->Write(spec) ? out : std::string();
}

// A Hydra dome texture: lat-long with +Y up in light space.
std::string _EqualAreaDome(std::string const &path, GfMatrix4d const &lightToWorld) {
    GfMatrix4d worldToLight = lightToWorld.GetInverse();
    return _EqualArea(path, TfStringify(lightToWorld), [&](GfVec3d const &dir) {
        GfVec3d d = worldToLight.TransformDir(dir).GetNormalized();
        return std::make_pair(1 - std::fmod(std::atan2(d[0], d[2]) / (2 * M_PI) + 0.5, 1.0),
                              std::acos(std::clamp(d[1], -1.0, 1.0)) / M_PI);
    });
}

// A world the Blender add-on baked: Blender's equirectangular convention, +Z up.
std::string _EqualAreaBlenderWorld(std::string const &path) {
    return _EqualArea(path, "blender", [](GfVec3d const &d) {
        return std::make_pair(-std::atan2(d[1], d[0]) / (2 * M_PI) + 0.5,
                              0.5 - std::atan2(d[2], std::hypot(d[0], d[1])) / M_PI);
    });
}

// Everything the prims synced, rebuilt into a Gonzales scene when it changes.
// A material the Blender add-on evaluated itself, for what UsdPreviewSurface cannot carry.
struct Override {
    TfToken kind;              // hair, hairlut, glass or basecolor
    GfVec3f sigma{0.0f};       // hair absorption
    float betaM = 0.3f, betaN = 0.3f, ior = 1.5f, alpha = 0.0f;
    std::string path;          // hairlut: colour tables; basecolor: the baked texture
    int intercepts = 0, randoms = 0, res = 0;
};

// Lines of "<material prim name> <kind> ..." as written by the add-on's materials.py.
std::map<std::string, Override> _ParseOverrides(std::string const &text) {
    std::map<std::string, Override> result;
    std::istringstream lines(text);
    std::string line, name, kind;
    while (std::getline(lines, line)) {
        std::istringstream in(line);
        Override o;
        if (!(in >> name >> kind))
            continue;
        o.kind = TfToken(kind);
        if (kind == "hair" && (in >> o.sigma[0] >> o.sigma[1] >> o.sigma[2] >> o.betaM >> o.betaN >> o.ior))
            result[name] = o;
        else if (kind == "hairlut" && (in >> o.path >> o.intercepts >> o.randoms >> o.res >> o.betaM >> o.betaN >> o.ior))
            result[name] = o;
        else if (kind == "glass" && (in >> o.ior >> o.alpha))
            result[name] = o;
        else if (kind == "basecolor" && (in >> o.path))
            result[name] = o;
    }
    return result;
}

class Scene final : public HdRenderParam {
public:
    std::string overrideText;
    std::map<std::string, Override> overrides;
    std::string worldPath, worldTexture;  // the add-on's baked world and its equal-area version
    HdRenderThread *thread;
    std::mutex mutex;
    std::map<SdfPath, MeshData> meshes;
    std::map<SdfPath, CurvesData> curves;
    std::map<SdfPath, MaterialData> materials;
    std::map<SdfPath, LightData> lights;
    std::atomic<int> version{1};

    template <typename F> void Edit(F f) {
        std::lock_guard<std::mutex> lock(mutex);
        f();
        version++;
    }
};

Scene &_Scene(HdRenderParam *param) { return *static_cast<Scene *>(param); }

// A primvar's values at each triangle corner, whatever its interpolation.
template <typename T>
bool _Corners(HdSceneDelegate *delegate, SdfPath const &id, HdMeshUtil &util, VtVec3iArray const &triangles,
              TfToken const &name, VtArray<T> *out) {
    for (HdInterpolation interpolation : {HdInterpolationVertex, HdInterpolationVarying, HdInterpolationFaceVarying})
        for (HdPrimvarDescriptor const &desc : delegate->GetPrimvarDescriptors(id, interpolation)) {
            if (desc.name != name)
                continue;
            VtValue value = delegate->Get(id, name);
            if (!value.IsHolding<VtArray<T>>())
                return false;
            VtArray<T> const &data = value.UncheckedGet<VtArray<T>>();
            if (interpolation == HdInterpolationFaceVarying) {
                VtValue corners;
                HdMeshComputationResult result = util.ComputeTriangulatedFaceVaryingPrimvar(
                    data.cdata(), int(data.size()), HdGetValueTupleType(value).type, &corners);
                if (result == HdMeshComputationResult::Unchanged)
                    *out = data;
                else if (result == HdMeshComputationResult::Success && corners.IsHolding<VtArray<T>>())
                    *out = corners.UncheckedGet<VtArray<T>>();
                else
                    return false;
                return out->size() == 3 * triangles.size();
            }
            out->resize(3 * triangles.size());
            for (size_t t = 0; t < triangles.size(); ++t)
                for (int k = 0; k < 3; ++k) {
                    int i = triangles[t][k];
                    if (i < 0 || size_t(i) >= data.size())
                        return false;
                    (*out)[3 * t + k] = data[i];
                }
            return true;
        }
    return false;
}

// "st", or else the first texture-coordinate primvar.
TfToken _UvName(HdSceneDelegate *delegate, SdfPath const &id) {
    TfToken found;
    for (HdInterpolation interpolation : {HdInterpolationVertex, HdInterpolationVarying, HdInterpolationFaceVarying})
        for (HdPrimvarDescriptor const &desc : delegate->GetPrimvarDescriptors(id, interpolation)) {
            if (desc.name == TfToken("st"))
                return desc.name;
            if (found.IsEmpty() && desc.role == HdPrimvarRoleTokens->textureCoordinate)
                found = desc.name;
        }
    return found;
}

class Mesh final : public HdMesh {
public:
    explicit Mesh(SdfPath const &id) : HdMesh(id) {}

    HdDirtyBits GetInitialDirtyBitsMask() const override {
        return HdChangeTracker::AllSceneDirtyBits;
    }

    void Sync(HdSceneDelegate *delegate, HdRenderParam *param, HdDirtyBits *dirtyBits,
              TfToken const &) override {
        SdfPath const &id = GetId();
        *dirtyBits &= ~HdChangeTracker::AllSceneDirtyBits;
        if (!delegate->GetVisible(id)) {
            _Scene(param).Edit([&] { _Scene(param).meshes.erase(id); });
            return;
        }
        MeshData m;
        HdMeshTopology topology = GetMeshTopology(delegate);
        HdMeshUtil util(&topology, id);
        VtVec3iArray triangles;
        VtIntArray primitiveParams;
        util.ComputeTriangleIndices(&triangles, &primitiveParams);
        size_t corners = 3 * triangles.size();
        VtVec3fArray points, normals;
        VtVec2fArray uvs;
        if (!_Corners(delegate, id, util, triangles, HdTokens->points, &points))
            return;
        m.points.assign(points.cdata()->data(), points.cdata()->data() + 3 * corners);
        m.indices.resize(corners);
        std::iota(m.indices.begin(), m.indices.end(), 0);
        GfMatrix4d xform = delegate->GetTransform(id);
        m.xform = GfMatrix4f(xform);
        if (_Corners(delegate, id, util, triangles, HdTokens->normals, &normals)) {
            GfMatrix4d normalXform = xform.GetInverse().GetTranspose();
            for (GfVec3f const &n : normals) {
                GfVec3f w = GfVec3f(normalXform.TransformDir(GfVec3d(n))).GetNormalized();
                m.normals.insert(m.normals.end(), {w[0], w[1], w[2]});
            }
        }
        if (_Corners(delegate, id, util, triangles, _UvName(delegate, id), &uvs))
            m.uvs.assign(uvs.cdata()->data(), uvs.cdata()->data() + 2 * corners);
        m.material = delegate->GetMaterialId(id);
        VtValue color = GetPrimvar(delegate, HdTokens->displayColor);
        if (color.IsHolding<VtVec3fArray>() && !color.UncheckedGet<VtVec3fArray>().empty())
            m.displayColor = color.UncheckedGet<VtVec3fArray>()[0];
        _Scene(param).Edit([&] { _Scene(param).meshes[id] = std::move(m); });
    }

    void Finalize(HdRenderParam *param) override {
        _Scene(param).Edit([&] { _Scene(param).meshes.erase(GetId()); });
    }

protected:
    HdDirtyBits _PropagateDirtyBits(HdDirtyBits bits) const override { return bits; }

    void _InitRepr(TfToken const &reprToken, HdDirtyBits *) override {
        if (std::find_if(_reprs.begin(), _reprs.end(), _ReprComparator(reprToken)) == _reprs.end())
            _reprs.emplace_back(reprToken, HdReprSharedPtr());
    }
};

// A per-strand random number in [0, 1), as Blender's Hair Info Random.
float _Random(uint32_t i) {
    i ^= i >> 16; i *= 0x7feb352d; i ^= i >> 15; i *= 0x846ca68b; i ^= i >> 16;
    return float(i >> 8) * (1.0f / 16777216.0f);
}

// Hydra curves as B-spline segments: Catmull-Rom and Bezier via Bezier control points.
class Curves final : public HdBasisCurves {
public:
    explicit Curves(SdfPath const &id) : HdBasisCurves(id) {}

    HdDirtyBits GetInitialDirtyBitsMask() const override { return HdChangeTracker::AllSceneDirtyBits; }

    void Sync(HdSceneDelegate *delegate, HdRenderParam *param, HdDirtyBits *dirtyBits, TfToken const &) override {
        SdfPath const &id = GetId();
        *dirtyBits &= ~HdChangeTracker::AllSceneDirtyBits;
        CurvesData c;
        HdBasisCurvesTopology topology = GetBasisCurvesTopology(delegate);
        VtVec3fArray points = delegate->Get(id, HdTokens->points).GetWithDefault<VtVec3fArray>();
        VtIntArray counts = topology.GetCurveVertexCounts();
        VtFloatArray widths;
        HdInterpolation widthInterpolation = HdInterpolationConstant;
        for (HdInterpolation interpolation : {HdInterpolationConstant, HdInterpolationVertex, HdInterpolationVarying})
            for (HdPrimvarDescriptor const &desc : delegate->GetPrimvarDescriptors(id, interpolation))
                if (desc.name == HdTokens->widths) {
                    VtValue value = delegate->Get(id, HdTokens->widths);
                    if (value.IsHolding<VtFloatArray>()) {
                        widths = value.UncheckedGet<VtFloatArray>();
                        widthInterpolation = interpolation;
                    }
                }
        bool visible = delegate->GetVisible(id) && topology.GetCurveIndices().empty() && !points.empty();
        if (visible) {
            GfMatrix4d xform = delegate->GetTransform(id);
            float scale = float(std::cbrt(std::abs(xform.ExtractRotationMatrix().GetDeterminant())));
            TfToken type = topology.GetCurveType(), basis = topology.GetCurveBasis(), wrap = topology.GetCurveWrap();
            size_t first = 0, firstVarying = 0, curve = 0;
            VtValue st = GetPrimvar(delegate, TfToken("st"));
            VtVec2fArray uvs = st.IsHolding<VtVec2fArray>() ? st.UncheckedGet<VtVec2fArray>() : VtVec2fArray();
            std::vector<GfVec3f> cv;
            std::vector<float> w;
            for (int count : counts) {
                cv.assign(points.begin() + first, points.begin() + first + count);
                int nSegments = _Segments(type, basis, wrap, count);
                w.clear();
                for (int i = 0; i <= nSegments; ++i) {
                    size_t k = widthInterpolation == HdInterpolationVertex ? first + std::min(count - 1, i)
                             : widthInterpolation == HdInterpolationVarying ? firstVarying + i : 0;
                    w.push_back((k < widths.size() ? widths[k] : 0.01f) * scale);
                }
                for (int i = 0; i < nSegments; ++i) {
                    GfVec3f b[4];
                    _Bezier(type, basis, wrap, cv, i, b);
                    GfVec3f bs[4] = {6 * b[0] - 7 * b[1] + 2 * b[2], 2 * b[1] - b[2], 2 * b[2] - b[1],
                                     2 * b[1] - 7 * b[2] + 6 * b[3]};
                    for (GfVec3f const &q : bs) {
                        GfVec3f p(xform.Transform(GfVec3d(q)));
                        c.controlPoints.insert(c.controlPoints.end(), {p[0], p[1], p[2]});
                    }
                    c.widths.insert(c.widths.end(), {w[i], w[i + 1]});
                    // Cycles reads a missing surface UV attribute as zero.
                    GfVec2f uv = curve < uvs.size() ? uvs[curve] : GfVec2f(0.0f);
                    c.lookup.insert(c.lookup.end(), {uv[0], uv[1], (i + 0.5f) / nSegments, _Random(curve)});
                }
                ++curve;
                first += count;
                firstVarying += nSegments + 1;
            }
        }
        c.material = delegate->GetMaterialId(id);
        VtValue color = GetPrimvar(delegate, HdTokens->displayColor);
        if (color.IsHolding<VtVec3fArray>() && !color.UncheckedGet<VtVec3fArray>().empty())
            c.displayColor = color.UncheckedGet<VtVec3fArray>()[0];
        _Scene(param).Edit([&] {
            if (c.widths.empty())
                _Scene(param).curves.erase(id);
            else
                _Scene(param).curves[id] = std::move(c);
        });
    }

    void Finalize(HdRenderParam *param) override {
        _Scene(param).Edit([&] { _Scene(param).curves.erase(GetId()); });
    }

protected:
    HdDirtyBits _PropagateDirtyBits(HdDirtyBits bits) const override { return bits; }

    void _InitRepr(TfToken const &reprToken, HdDirtyBits *) override {
        if (std::find_if(_reprs.begin(), _reprs.end(), _ReprComparator(reprToken)) == _reprs.end())
            _reprs.emplace_back(reprToken, HdReprSharedPtr());
    }

private:
    static int _Segments(TfToken const &type, TfToken const &basis, TfToken const &wrap, int count) {
        if (type == HdTokens->linear)
            return std::max(0, count - 1);
        if (basis == HdTokens->bezier)
            return std::max(0, (count - 1) / 3);
        if (wrap == HdTokens->pinned)
            return std::max(0, count - 1);
        return std::max(0, count - 3);
    }

    // Bezier control points of segment i.
    static void _Bezier(TfToken const &type, TfToken const &basis, TfToken const &wrap,
                        std::vector<GfVec3f> const &cv, int i, GfVec3f b[4]) {
        int n = int(cv.size());
        if (type == HdTokens->linear) {
            b[0] = cv[i]; b[3] = cv[i + 1];
            b[1] = b[0] + (b[3] - b[0]) / 3; b[2] = b[0] + 2 * (b[3] - b[0]) / 3;
            return;
        }
        if (basis == HdTokens->bezier) {
            for (int k = 0; k < 4; ++k)
                b[k] = cv[3 * i + k];
            return;
        }
        // Pinned curves get phantom end points that make them pass through their ends.
        auto p = [&](int k) {
            if (wrap != HdTokens->pinned)
                return cv[k];
            if (k == 0)
                return 2 * cv[0] - cv[std::min(1, n - 1)];
            if (k == n + 1)
                return 2 * cv[n - 1] - cv[std::max(0, n - 2)];
            return cv[k - 1];
        };
        GfVec3f p0 = p(i), p1 = p(i + 1), p2 = p(i + 2), p3 = p(i + 3);
        if (basis == HdTokens->bSpline) {
            b[0] = (p0 + 4 * p1 + p2) / 6; b[1] = (2 * p1 + p2) / 3;
            b[2] = (p1 + 2 * p2) / 3;      b[3] = (p1 + 4 * p2 + p3) / 6;
            return;
        }
        b[0] = p1; b[1] = p1 + (p2 - p0) / 6; b[2] = p2 - (p3 - p1) / 6; b[3] = p2;
    }
};

// An image as top-down RGB rows, read once per path.
struct Image {
    int width = 0, height = 0;
    std::vector<GfVec3f> pixels;
};

Image const &_ReadImage(std::string const &path) {
    static std::mutex mutex;
    static std::map<std::string, Image> cache;
    std::lock_guard<std::mutex> lock(mutex);
    auto [it, inserted] = cache.try_emplace(path);
    if (inserted && path.size() > 4 && path.compare(path.size() - 4, 4, ".pfm") == 0) {
        // Hio has no PFM reader: RGB floats, rows bottom to top, little endian when the scale is negative.
        std::ifstream in(path, std::ios::binary);
        std::string magic;
        int w = 0, h = 0;
        float scale = 0;
        if (in >> magic >> w >> h >> scale && magic == "PF" && scale < 0 && w > 0 && h > 0) {
            in.get();
            std::vector<GfVec3f> rows(size_t(w) * h);
            if (in.read(reinterpret_cast<char *>(rows.data()), std::streamsize(rows.size() * sizeof(GfVec3f)))) {
                Image &image = it->second;
                image.width = w;
                image.height = h;
                image.pixels.resize(rows.size());
                for (int y = 0; y < h; ++y)
                    std::copy_n(&rows[size_t(h - 1 - y) * w], w, &image.pixels[size_t(y) * w]);
            }
        }
    } else if (inserted) {
        if (HioImageSharedPtr in = HioImage::OpenForReading(path)) {
            Image &image = it->second;
            image.width = in->GetWidth();
            image.height = in->GetHeight();
            image.pixels.resize(size_t(image.width) * image.height);
            HioImage::StorageSpec spec;
            spec.width = image.width; spec.height = image.height; spec.depth = 1;
            spec.format = HioFormatFloat32Vec3;
            spec.data = image.pixels.data();
            if (!in->Read(spec))
                image = Image();
        }
    }
    return it->second;
}

// pbrt-v4's HairBxDF::SigmaAFromReflectance.
GfVec3f _SigmaAFromReflectance(GfVec3f const &c, float betaN) {
    float d = 5.969f - 0.215f * betaN + 2.532f * betaN * betaN - 10.73f * std::pow(betaN, 3.0f) +
              5.574f * std::pow(betaN, 4.0f) + 0.245f * std::pow(betaN, 5.0f);
    GfVec3f sigma;
    for (int i = 0; i < 3; ++i) {
        float l = std::log(std::clamp(c[i], 1e-4f, 1.0f)) / d;
        sigma[i] = l * l;
    }
    return sigma;
}

// UsdPreviewSurface's diffuse colour or texture and its normal map.
class Material final : public HdMaterial {
public:
    explicit Material(SdfPath const &id) : HdMaterial(id) {}

    HdDirtyBits GetInitialDirtyBitsMask() const override { return AllDirty; }

    void Sync(HdSceneDelegate *delegate, HdRenderParam *param, HdDirtyBits *dirtyBits) override {
        *dirtyBits = Clean;
        MaterialData m;
        VtValue resource = delegate->GetMaterialResource(GetId());
        if (resource.IsHolding<HdMaterialNetworkMap>()) {
            HdMaterialNetwork2 network = HdConvertToHdMaterialNetwork2(resource.UncheckedGet<HdMaterialNetworkMap>());
            auto terminal = network.terminals.find(HdMaterialTerminalTokens->surface);
            auto surface = terminal == network.terminals.end() ? network.nodes.end()
                                                                : network.nodes.find(terminal->second.upstreamNode);
            if (surface != network.nodes.end() && surface->second.nodeTypeId == TfToken("UsdPreviewSurface")) {
                HdMaterialNode2 const &node = surface->second;
                auto texture = [&](char const *input) -> std::string {
                    auto c = node.inputConnections.find(TfToken(input));
                    if (c == node.inputConnections.end() || c->second.empty())
                        return {};
                    auto upstream = network.nodes.find(c->second[0].upstreamNode);
                    if (upstream == network.nodes.end() || upstream->second.nodeTypeId != TfToken("UsdUVTexture"))
                        return {};
                    auto file = upstream->second.parameters.find(TfToken("file"));
                    if (file == upstream->second.parameters.end() || !file->second.IsHolding<SdfAssetPath>())
                        return {};
                    SdfAssetPath const &asset = file->second.UncheckedGet<SdfAssetPath>();
                    return asset.GetResolvedPath().empty() ? asset.GetAssetPath() : asset.GetResolvedPath();
                };
                auto diffuse = node.parameters.find(TfToken("diffuseColor"));
                if (diffuse != node.parameters.end() && diffuse->second.IsHolding<GfVec3f>())
                    m.diffuse = diffuse->second.UncheckedGet<GfVec3f>();
                m.texture = texture("diffuseColor");
                m.normalMap = texture("normal");
                if (!m.texture.empty())
                    m.diffuse = GfVec3f(1);
            }
        }
        _Scene(param).Edit([&] { _Scene(param).materials[GetId()] = std::move(m); });
    }

    void Finalize(HdRenderParam *param) override {
        _Scene(param).Edit([&] { _Scene(param).materials.erase(GetId()); });
    }
};

// UsdLux rect and disk lights lie in their XY plane and emit along -Z.
void _AreaLightMesh(HdSceneDelegate *delegate, SdfPath const &id, LightData &light) {
    auto param = [&](char const *name, float fallback) {
        return delegate->GetLightParamValue(id, TfToken(name)).GetWithDefault<float>(fallback);
    };
    if (light.type == HdPrimTypeTokens->rectLight) {
        float w = param("width", 1) / 2, h = param("height", 1) / 2;
        light.points = {-w, -h, 0, w, -h, 0, w, h, 0, -w, h, 0};
        light.indices = {0, 2, 1, 0, 3, 2};
        return;
    }
    float r = param("radius", 0.5f);
    int n = 64;
    light.points = {0, 0, 0};
    for (int i = 0; i < n; ++i) {
        double phi = 2 * M_PI * i / n;
        light.points.insert(light.points.end(), {float(r * std::cos(phi)), float(r * std::sin(phi)), 0});
        light.indices.insert(light.indices.end(), {0, 1 + (i + 1) % n, 1 + i});
    }
}

// World-space area of an emitting mesh.
float _Area(LightData const &light) {
    auto p = [&](int i) {
        int k = light.indices[i];
        return GfVec3d(light.xform.Transform(GfVec3f(light.points[3 * k], light.points[3 * k + 1], light.points[3 * k + 2])));
    };
    double area = 0;
    for (size_t t = 0; t < light.indices.size(); t += 3)
        area += 0.5 * GfCross(p(t + 1) - p(t), p(t + 2) - p(t)).GetLength();
    return float(area);
}

class Light final : public HdLight {
public:
    Light(SdfPath const &id, TfToken const &type) : HdLight(id), _type(type) {}

    HdDirtyBits GetInitialDirtyBitsMask() const override { return AllDirty; }

    void Sync(HdSceneDelegate *delegate, HdRenderParam *param, HdDirtyBits *dirtyBits) override {
        *dirtyBits = Clean;
        SdfPath const &id = GetId();
        GfVec3f color = delegate->GetLightParamValue(id, HdLightTokens->color).GetWithDefault<GfVec3f>(GfVec3f(1));
        float intensity = delegate->GetLightParamValue(id, HdLightTokens->intensity).GetWithDefault<float>(1);
        float exposure = delegate->GetLightParamValue(id, HdLightTokens->exposure).GetWithDefault<float>(0);
        GfMatrix4d xform = delegate->GetTransform(id);
        bool normalize = delegate->GetLightParamValue(id, HdLightTokens->normalize).GetWithDefault<bool>(false);
        GfVec3f rgb = color * intensity * std::exp2(exposure);
        if (delegate->GetLightParamValue(id, TfToken("enableColorTemperature")).GetWithDefault<bool>(false))
            rgb = GfCompMult(rgb, UsdLuxBlackbodyTemperatureAsRgb(
                delegate->GetLightParamValue(id, TfToken("colorTemperature")).GetWithDefault<float>(6500)));
        // Lights shine along their -Z axis.
        LightData light{_type, rgb, -GfVec3f(xform.GetRow3(2)).GetNormalized(), GfVec3f(xform.ExtractTranslation()), 0, {},
                        GfMatrix4f(xform), {}, {}};
        if (_type == HdPrimTypeTokens->distantLight && !normalize) {
            // UsdLux: intensity is radiance unless normalized to illuminance.
            double theta = std::clamp(delegate->GetLightParamValue(id, HdLightTokens->angle).GetWithDefault<float>(0.53f) * M_PI / 360, 0.0, M_PI);
            double s2 = std::sin(theta) * std::sin(theta);
            light.rgb *= float(theta == 0 ? 1 : (theta <= M_PI / 2 ? s2 : 2 - s2) * M_PI);
        } else if (_type == HdPrimTypeTokens->sphereLight) {
            float scale = float(std::cbrt(std::abs(xform.ExtractRotationMatrix().GetDeterminant())));
            light.radius = delegate->GetLightParamValue(id, HdLightTokens->radius).GetWithDefault<float>(0.5f) * scale;
            if (delegate->GetLightParamValue(id, TfToken("treatAsPoint")).GetWithDefault<bool>(false))
                light.radius = 0;
            if (light.radius == 0)
                light.rgb /= 4;  // the radiant intensity of a normalized sphere
            else if (normalize)
                light.rgb /= float(4 * M_PI * light.radius * light.radius);
        } else if (_type == HdPrimTypeTokens->rectLight || _type == HdPrimTypeTokens->diskLight) {
            _AreaLightMesh(delegate, id, light);
            if (normalize)
                light.rgb /= _Area(light);
        }
        VtValue texture = delegate->GetLightParamValue(id, HdLightTokens->textureFile);
        if (_type == HdPrimTypeTokens->domeLight && texture.IsHolding<SdfAssetPath>()) {
            SdfAssetPath const &asset = texture.UncheckedGet<SdfAssetPath>();
            std::string file = asset.GetResolvedPath().empty() ? asset.GetAssetPath() : asset.GetResolvedPath();
            if (!file.empty())
                light.texture = _EqualAreaDome(file, xform);
        }
        _Scene(param).Edit([&] { _Scene(param).lights[id] = light; });
    }

    void Finalize(HdRenderParam *param) override {
        _Scene(param).Edit([&] { _Scene(param).lights.erase(GetId()); });
    }

private:
    TfToken _type;
};

void _Store(HdFormat format, uint8_t *dst, float const *value, size_t n) {
    HdFormat component = HdGetComponentFormat(format);
    for (size_t c = 0; c < HdGetComponentCount(format); ++c) {
        float v = c < n ? value[c] : 0.0f;
        switch (component) {
        case HdFormatFloat32: reinterpret_cast<float *>(dst)[c] = v; break;
        case HdFormatFloat16: reinterpret_cast<uint16_t *>(dst)[c] = GfHalf(v).bits(); break;
        case HdFormatUNorm8: dst[c] = uint8_t(std::clamp(v, 0.0f, 1.0f) * 255.0f + 0.5f); break;
        case HdFormatInt32: reinterpret_cast<int32_t *>(dst)[c] = int32_t(v); break;
        default: break;
        }
    }
}

std::vector<float> _Floats(VtValue const &v) {
    if (v.IsHolding<GfVec4f>()) { GfVec4f x = v.UncheckedGet<GfVec4f>(); return {x[0], x[1], x[2], x[3]}; }
    if (v.IsHolding<GfVec3f>()) { GfVec3f x = v.UncheckedGet<GfVec3f>(); return {x[0], x[1], x[2]}; }
    if (v.IsHolding<float>()) return {v.UncheckedGet<float>()};
    if (v.IsHolding<int>()) return {float(v.UncheckedGet<int>())};
    return {};
}

class RenderBuffer final : public HdRenderBuffer {
public:
    explicit RenderBuffer(SdfPath const &id) : HdRenderBuffer(id) {}

    // The render thread writes into the buffer: stop it before reallocating.
    void Sync(HdSceneDelegate *delegate, HdRenderParam *param, HdDirtyBits *dirtyBits) override {
        if (*dirtyBits & DirtyDescription)
            _Scene(param).thread->StopRender();
        HdRenderBuffer::Sync(delegate, param, dirtyBits);
    }
    void Finalize(HdRenderParam *param) override {
        _Scene(param).thread->StopRender();
        HdRenderBuffer::Finalize(param);
    }

    bool Allocate(GfVec3i const &dims, HdFormat format, bool) override {
        _Deallocate();
        _width = dims[0];
        _height = dims[1];
        _format = format;
        _buffer.resize(size_t(_width) * _height * HdDataSizeOfFormat(format));
        return true;
    }
    unsigned int GetWidth() const override { return _width; }
    unsigned int GetHeight() const override { return _height; }
    unsigned int GetDepth() const override { return 1; }
    HdFormat GetFormat() const override { return _format; }
    bool IsMultiSampled() const override { return false; }
    void *Map() override { _mappers++; return _buffer.data(); }
    void Unmap() override { _mappers--; }
    bool IsMapped() const override { return _mappers.load() != 0; }
    void Resolve() override {}
    bool IsConverged() const override { return _converged.load(); }
    void SetConverged(bool converged) { _converged.store(converged); }

    // Row 0 of a Hydra buffer is the bottom of the image.
    void Store(unsigned x, unsigned y, float const *value, size_t n) {
        size_t size = HdDataSizeOfFormat(_format);
        _Store(_format, &_buffer[(size_t(_height - 1 - y) * _width + x) * size], value, n);
    }
    // width * height RGB values, top row first.
    void StoreRgb(float const *rgb) {
        for (unsigned y = 0; y < _height; ++y) {
            float const *src = rgb + size_t(y) * _width * 3;
            if (_format != HdFormatFloat32Vec4) {
                for (unsigned x = 0; x < _width; ++x) {
                    float value[4] = {src[3 * x], src[3 * x + 1], src[3 * x + 2], 1.0f};
                    Store(x, y, value, 4);
                }
                continue;
            }
            float *dst = reinterpret_cast<float *>(&_buffer[size_t(_height - 1 - y) * _width * 16]);
            for (unsigned x = 0; x < _width; ++x, dst += 4, src += 3) {
                dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2]; dst[3] = 1.0f;
            }
        }
    }
    void Fill(std::vector<float> const &value) {
        size_t size = HdDataSizeOfFormat(_format);
        for (size_t i = 0; i < size_t(_width) * _height; ++i)
            _Store(_format, &_buffer[i * size], value.data(), value.size());
    }

private:
    void _Deallocate() override {
        _width = _height = 0;
        _format = HdFormatInvalid;
        _buffer.clear();
    }

    unsigned _width = 0, _height = 0;
    HdFormat _format = HdFormatInvalid;
    std::vector<uint8_t> _buffer;
    std::atomic<int> _mappers{0};
    std::atomic<bool> _converged{true};
};

void *_Renderer() {
    static void *renderer = gz_renderer_new(TfGetenv("GONZALES_DATA_DIR", GONZALES_DATA_DIR).c_str());
    return renderer;
}

// The camera, image and samples of a render.
struct View {
    GfMatrix4f worldToCamera{1.0f};
    float fov = 0;
    int width = 0, height = 0, spp = 0, maxDepth = 0;
    RenderBuffer *color = nullptr;

    bool operator==(View const &o) const {
        return worldToCamera == o.worldToCamera && fov == o.fov && width == o.width && height == o.height &&
               spp == o.spp && maxDepth == o.maxDepth && color == o.color;
    }
};

// Per-segment hair materials from the add-on's colour tables: each segment's colour is quantized into a
// palette entry, created on first use.
std::vector<int> _HairPalette(void *scene, Override const &lut, std::vector<float> const &lookup,
                              std::map<int, int> &palette) {
    Image const &image = _ReadImage(lut.path);
    int r = lut.res;
    if (image.width != r * lut.randoms || image.height != r * lut.intercepts)
        return {};
    constexpr int levels = 24;
    std::vector<int> result(lookup.size() / 4);
    for (size_t s = 0; s < result.size(); ++s) {
        float const *l = &lookup[4 * s];
        int i = std::clamp(int(l[2] * lut.intercepts), 0, lut.intercepts - 1);
        int k = std::clamp(int(l[3] * lut.randoms), 0, lut.randoms - 1);
        int x = std::clamp(int((l[0] - std::floor(l[0])) * r), 0, r - 1);
        int y = std::clamp(int((l[1] - std::floor(l[1])) * r), 0, r - 1);
        // The table's rows go up in v and intercept; the image is read top-down.
        GfVec3f c = image.pixels[size_t(image.height - 1 - (i * r + y)) * image.width + k * r + x];
        int key = 0;
        for (int ch = 0; ch < 3; ++ch)
            key = key * levels + std::clamp(int(std::sqrt(std::clamp(c[ch], 0.0f, 1.0f)) * levels), 0, levels - 1);
        auto it = palette.find(key);
        if (it == palette.end()) {
            GfVec3f q;
            for (int ch = 2, k2 = key; ch >= 0; --ch, k2 /= levels) {
                float v = (k2 % levels + 0.5f) / levels;
                q[ch] = v * v;
            }
            GfVec3f sigma = _SigmaAFromReflectance(q, lut.betaN);
            it = palette.emplace(key, gz_scene_add_hair(scene, sigma[0], sigma[1], sigma[2], lut.betaM, lut.betaN, lut.ior)).first;
        }
        result[s] = it->second;
    }
    return result;
}

void *_BuildScene(Scene &synced, int *version) {
    void *scene = gz_scene_new();
    std::lock_guard<std::mutex> lock(synced.mutex);
    *version = synced.version.load();
    std::map<SdfPath, int> materials;
    auto path = [](std::string const &p) { return p.empty() ? nullptr : p.c_str(); };
    auto override = [&](SdfPath const &id, TfToken const &kind) -> Override const * {
        auto it = synced.overrides.find(id.GetName());
        return it != synced.overrides.end() && it->second.kind == kind ? &it->second : nullptr;
    };
    for (auto const &[id, mat] : synced.materials)
        if (Override const *glass = override(id, TfToken("glass")))
            materials[id] = gz_scene_add_dielectric(scene, glass->ior, glass->alpha);
        else if (Override const *base = override(id, TfToken("basecolor")))
            materials[id] = gz_scene_add_diffuse(scene, 1, 1, 1, base->path.c_str(), path(mat.normalMap));
        else
            materials[id] = gz_scene_add_diffuse(scene, mat.diffuse[0], mat.diffuse[1], mat.diffuse[2],
                                                 path(mat.texture), path(mat.normalMap));
    for (auto const &[id, m] : synced.meshes) {
        if (m.indices.empty())
            continue;
        // A glass-only Blender material has no Hydra material, only the add-on's override.
        if (!materials.count(m.material))
            if (Override const *glass = override(m.material, TfToken("glass")))
                materials[m.material] = gz_scene_add_dielectric(scene, glass->ior, glass->alpha);
        auto it = materials.find(m.material);
        int material = it != materials.end() ? it->second
            : gz_scene_add_diffuse(scene, m.displayColor[0], m.displayColor[1], m.displayColor[2], nullptr, nullptr);
        gz_scene_add_mesh(scene, m.xform.data(), m.points.data(), int(m.points.size() / 3),
                          m.indices.data(), int(m.indices.size() / 3),
                          m.normals.empty() ? nullptr : m.normals.data(),
                          m.uvs.empty() ? nullptr : m.uvs.data(), material);
    }
    std::map<SdfPath, int> hairs;
    std::map<SdfPath, std::map<int, int>> palettes;
    for (auto const &[id, c] : synced.curves) {
        if (!hairs.count(c.material)) {
            Override h;
            if (Override const *o = override(c.material, TfToken("hair"))) {
                h = *o;
            } else {
                auto it = synced.materials.find(c.material);
                h.sigma = _SigmaAFromReflectance(it == synced.materials.end() ? c.displayColor
                                                 : it->second.texture.empty() ? it->second.diffuse : GfVec3f(0.5f), h.betaN);
                h.ior = 1.55f;
            }
            hairs[c.material] = gz_scene_add_hair(scene, h.sigma[0], h.sigma[1], h.sigma[2], h.betaM, h.betaN, h.ior);
        }
        std::vector<int> perSegment;
        if (Override const *lut = override(c.material, TfToken("hairlut")))
            perSegment = _HairPalette(scene, *lut, c.lookup, palettes[c.material]);
        gz_scene_add_curves(scene, c.controlPoints.data(), c.widths.data(), int(c.widths.size() / 2),
                            hairs[c.material], perSegment.empty() ? nullptr : perSegment.data());
    }
    bool lit = false;
    if (!synced.worldTexture.empty()) {
        gz_scene_add_dome_map(scene, synced.worldTexture.c_str(), 1, 1, 1);
        lit = true;
    }
    for (auto const &[id, l] : synced.lights) {
        if (l.type == HdPrimTypeTokens->domeLight && !synced.worldTexture.empty())
            continue;
        if (l.type == HdPrimTypeTokens->domeLight && !l.texture.empty())
            gz_scene_add_dome_map(scene, l.texture.c_str(), l.rgb[0], l.rgb[1], l.rgb[2]);
        else if (l.type == HdPrimTypeTokens->domeLight)
            gz_scene_add_dome(scene, l.rgb[0], l.rgb[1], l.rgb[2]);
        else if (l.type == HdPrimTypeTokens->distantLight)
            gz_scene_add_distant(scene, l.dir[0], l.dir[1], l.dir[2], l.rgb[0], l.rgb[1], l.rgb[2]);
        else if (!l.indices.empty())
            gz_scene_add_area_light(scene, l.xform.data(), l.points.data(), int(l.points.size() / 3),
                                    l.indices.data(), int(l.indices.size() / 3), l.rgb[0], l.rgb[1], l.rgb[2]);
        else if (l.type == HdPrimTypeTokens->sphereLight && l.radius == 0)
            gz_scene_add_point(scene, l.pos[0], l.pos[1], l.pos[2], l.rgb[0], l.rgb[1], l.rgb[2]);
        else if (l.type == HdPrimTypeTokens->sphereLight)
            gz_scene_add_sphere_light(scene, l.pos[0], l.pos[1], l.pos[2], l.radius, l.rgb[0], l.rgb[1], l.rgb[2]);
        else
            continue;
        lit = true;
    }
    if (!lit)
        gz_scene_add_dome(scene, 1, 1, 1);
    return scene;
}

// Progressive rendering on the render thread. The session keeps the scene on the GPU until the scene changes;
// the render pass sets the view only while the thread is stopped.
class Renderer {
public:
    explicit Renderer(Scene *scene) : _scene(scene) {}
    ~Renderer() { Free(); }

    View view;
    std::atomic<int> spp{0};

    void Free() {
        if (_session)
            gz_session_free(_session);
        _session = nullptr;
    }

    void Render(HdRenderThread &thread) {
        View v = view;
        spp = 0;
        if (!v.color || v.width <= 0 || v.height <= 0)
            return;
        if (!_session || _version != _scene->version.load()) {
            Free();
            void *scene = _BuildScene(*_scene, &_version);
            gz_scene_set_camera(scene, v.worldToCamera.data(), v.fov, v.width, v.height);
            _session = gz_session_new(_Renderer(), scene, v.spp, v.maxDepth);
            if (!_session) {
                v.color->SetConverged(true);
                return;
            }
        }
        gz_session_set_samples(_session, v.spp, v.maxDepth);
        if (gz_session_set_camera(_session, v.worldToCamera.data(), v.fov, v.width, v.height) != 0) {
            v.color->SetConverged(true);
            return;
        }
        std::vector<float> rgb(size_t(v.width) * v.height * 3);
        bool failed = false;
        // Small passes first, for a quick first image.
        for (int pass = 1; !thread.IsStopRequested(); pass = std::min(2 * pass, 8)) {
            int done = gz_session_render(_session, pass);
            if ((failed = done < 0 || gz_session_resolve(_session, rgb.data(), 1) != 0))
                break;
            v.color->StoreRgb(rgb.data());
            spp = done;
            if (done >= v.spp)
                break;
        }
        if (failed || spp >= v.spp)
            v.color->SetConverged(true);
    }

private:
    Scene *_scene;
    void *_session = nullptr;
    int _version = 0;
};

// Restarts the render thread whenever the scene, camera, image or settings changed.
class RenderPass final : public HdRenderPass {
public:
    RenderPass(HdRenderIndex *index, HdRprimCollection const &collection, Scene *scene, Renderer *renderer,
               HdRenderThread *thread)
        : HdRenderPass(index, collection), _scene(scene), _renderer(renderer), _thread(thread) {}
    ~RenderPass() override { _thread->StopRender(); }

    bool IsConverged() const override { return !_color || _color->IsConverged(); }

protected:
    void _Execute(HdRenderPassStateSharedPtr const &state, TfTokenVector const &) override {
        RenderBuffer *color = nullptr;
        std::vector<std::pair<RenderBuffer *, VtValue>> others;
        for (HdRenderPassAovBinding const &binding : state->GetAovBindings()) {
            HdRenderBuffer *buffer = binding.renderBuffer ? binding.renderBuffer
                : static_cast<HdRenderBuffer *>(GetRenderIndex()->GetBprim(HdPrimTypeTokens->renderBuffer, binding.renderBufferId));
            if (!buffer)
                continue;
            if (binding.aovName == HdAovTokens->color)
                color = static_cast<RenderBuffer *>(buffer);
            else
                others.emplace_back(static_cast<RenderBuffer *>(buffer), binding.clearValue);
        }
        _color = color;
        if (!color || !_Renderer())
            return;
        View v;
        v.color = color;
        v.width = color->GetWidth();
        v.height = color->GetHeight();
        GfMatrix4d view = state->GetWorldToViewMatrix(), proj = state->GetProjectionMatrix();
        // Gonzales cameras look along +z: negate the view's z column.
        v.worldToCamera = GfMatrix4f(view);
        for (int r = 0; r < 4; ++r)
            v.worldToCamera[r][2] = -v.worldToCamera[r][2];
        double scale = v.width >= v.height ? proj[1][1] : proj[0][0];
        v.fov = float(2.0 * std::atan(1.0 / scale) * 180.0 / M_PI);
        HdRenderDelegate *delegate = GetRenderIndex()->GetRenderDelegate();
        v.spp = std::max(1, delegate->GetRenderSetting<int>(HdRenderSettingsTokens->convergedSamplesPerPixel, 64));
        v.maxDepth = delegate->GetRenderSetting<int>(TfToken("maxDepth"), 5);
        std::string overrides = delegate->GetRenderSetting<std::string>(TfToken("gonzales:materials"), std::string());
        std::string world = delegate->GetRenderSetting<std::string>(TfToken("gonzales:world"), std::string());
        if (world != _scene->worldPath) {
            _thread->StopRender();
            std::string texture = world.empty() ? std::string() : _EqualAreaBlenderWorld(world);
            _scene->Edit([&] {
                _scene->worldPath = world;
                _scene->worldTexture = texture;
            });
        }
        if (overrides != _scene->overrideText) {
            _thread->StopRender();
            _scene->Edit([&] {
                _scene->overrideText = overrides;
                _scene->overrides = _ParseOverrides(overrides);
            });
        }
        int version = _scene->version.load();
        if (version == _version && v == _renderer->view)
            return;
        _version = version;
        _thread->StopRender();
        _renderer->view = v;
        _renderer->spp = 0;
        color->SetConverged(false);
        for (auto &[buffer, clear] : others)
            buffer->Fill(_Floats(clear));
        _thread->StartRender();
    }

private:
    Scene *_scene;
    Renderer *_renderer;
    HdRenderThread *_thread;
    RenderBuffer *_color = nullptr;
    int _version = 0;
};

class RenderDelegate final : public HdRenderDelegate {
public:
    explicit RenderDelegate(HdRenderSettingsMap const &settings = {})
        : HdRenderDelegate(settings), _registry(std::make_shared<HdResourceRegistry>()), _renderer(&_scene) {
        _settingDescriptors = {
            {"Samples per pixel", HdRenderSettingsTokens->convergedSamplesPerPixel, VtValue(64)},
            {"Max depth", TfToken("maxDepth"), VtValue(5)},
        };
        _PopulateDefaultSettings(_settingDescriptors);
        _scene.thread = &_thread;
        _thread.SetRenderCallback([this] { _renderer.Render(_thread); });
        _thread.StartThread();
    }
    ~RenderDelegate() override {
        _thread.StopThread();
        _renderer.Free();
    }

    VtDictionary GetRenderStats() const override {
        int spp = _renderer.view.spp;
        return {{"percentDone", VtValue(spp > 0 ? 100.0 * _renderer.spp.load() / spp : 0.0)}};
    }

    HdRenderParam *GetRenderParam() const override { return &_scene; }

    TfTokenVector const &GetSupportedRprimTypes() const override {
        static TfTokenVector const types = {HdPrimTypeTokens->mesh, HdPrimTypeTokens->basisCurves};
        return types;
    }
    TfTokenVector const &GetSupportedSprimTypes() const override {
        static TfTokenVector const types = {HdPrimTypeTokens->camera, HdPrimTypeTokens->material,
                                            HdPrimTypeTokens->domeLight, HdPrimTypeTokens->distantLight,
                                            HdPrimTypeTokens->sphereLight, HdPrimTypeTokens->rectLight,
                                            HdPrimTypeTokens->diskLight};
        return types;
    }
    TfTokenVector const &GetSupportedBprimTypes() const override {
        static TfTokenVector const types = {HdPrimTypeTokens->renderBuffer};
        return types;
    }
    HdResourceRegistrySharedPtr GetResourceRegistry() const override { return _registry; }
    HdRenderSettingDescriptorList GetRenderSettingDescriptors() const override { return _settingDescriptors; }

    HdAovDescriptor GetDefaultAovDescriptor(TfToken const &name) const override {
        if (name == HdAovTokens->color)
            return HdAovDescriptor(HdFormatFloat32Vec4, false, VtValue(GfVec4f(0, 0, 0, 1)));
        if (name == HdAovTokens->depth)
            return HdAovDescriptor(HdFormatFloat32, false, VtValue(1.0f));
        if (name == HdAovTokens->primId || name == HdAovTokens->instanceId || name == HdAovTokens->elementId)
            return HdAovDescriptor(HdFormatInt32, false, VtValue(-1));
        return HdAovDescriptor();
    }

    HdRenderPassSharedPtr CreateRenderPass(HdRenderIndex *index, HdRprimCollection const &collection) override {
        return std::make_shared<RenderPass>(index, collection, &_scene, &_renderer, &_thread);
    }
    HdInstancer *CreateInstancer(HdSceneDelegate *, SdfPath const &) override { return nullptr; }
    void DestroyInstancer(HdInstancer *instancer) override { delete instancer; }

    HdRprim *CreateRprim(TfToken const &type, SdfPath const &id) override {
        if (type == HdPrimTypeTokens->mesh)
            return new Mesh(id);
        if (type == HdPrimTypeTokens->basisCurves)
            return new Curves(id);
        return nullptr;
    }
    void DestroyRprim(HdRprim *prim) override { delete prim; }

    HdSprim *CreateSprim(TfToken const &type, SdfPath const &id) override {
        if (type == HdPrimTypeTokens->camera)
            return new HdCamera(id);
        if (type == HdPrimTypeTokens->material)
            return new Material(id);
        if (type == HdPrimTypeTokens->domeLight || type == HdPrimTypeTokens->distantLight ||
            type == HdPrimTypeTokens->sphereLight || type == HdPrimTypeTokens->rectLight ||
            type == HdPrimTypeTokens->diskLight)
            return new Light(id, type);
        return nullptr;
    }
    HdSprim *CreateFallbackSprim(TfToken const &type) override { return CreateSprim(type, SdfPath::EmptyPath()); }
    void DestroySprim(HdSprim *prim) override { delete prim; }

    HdBprim *CreateBprim(TfToken const &type, SdfPath const &id) override {
        return type == HdPrimTypeTokens->renderBuffer ? new RenderBuffer(id) : nullptr;
    }
    HdBprim *CreateFallbackBprim(TfToken const &type) override { return CreateBprim(type, SdfPath::EmptyPath()); }
    void DestroyBprim(HdBprim *prim) override {
        _thread.StopRender();
        delete prim;
    }

    void CommitResources(HdChangeTracker *) override {}

private:
    mutable Scene _scene;
    HdResourceRegistrySharedPtr _registry;
    HdRenderSettingDescriptorList _settingDescriptors;
    HdRenderThread _thread;
    Renderer _renderer;
};

} // namespace

class HdGonzalesRendererPlugin final : public HdRendererPlugin {
public:
    HdRenderDelegate *CreateRenderDelegate() override { return new RenderDelegate(); }
    HdRenderDelegate *CreateRenderDelegate(HdRenderSettingsMap const &settings) override {
        return new RenderDelegate(settings);
    }
    void DeleteRenderDelegate(HdRenderDelegate *delegate) override { delete delegate; }
    bool IsSupported(HdRendererCreateArgsSchema const &, std::string *) const override { return true; }
};

TF_REGISTRY_FUNCTION(TfType) {
    HdRendererPluginRegistry::Define<HdGonzalesRendererPlugin>();
}

PXR_NAMESPACE_CLOSE_SCOPE
