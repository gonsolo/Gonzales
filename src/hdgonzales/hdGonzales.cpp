// Hydra render delegate: collects meshes, materials, lights and the camera,
// and renders them with libgonzales.so.
#include "gonzales.h"

#include "pxr/base/gf/half.h"
#include "pxr/base/gf/matrix3d.h"
#include "pxr/base/gf/matrix4f.h"
#include "pxr/base/tf/getenv.h"
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
#include <map>
#include <mutex>
#include <vector>

PXR_NAMESPACE_OPEN_SCOPE

namespace {

struct MeshData {
    std::vector<float> points;
    std::vector<int> indices;
    GfMatrix4f xform;
    SdfPath material;
    GfVec3f displayColor{0.5f};
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

// Resamples a Hydra dome texture (lat-long, +Y up in light space) into an
// equal-area square in world space and returns the path of that image.
std::string _EqualAreaDome(std::string const &path, GfMatrix4d const &lightToWorld) {
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
    GfMatrix4d worldToLight = lightToWorld.GetInverse();
    for (int y = 0; y < n; ++y)
        for (int x = 0; x < n; ++x) {
            GfVec3d d = worldToLight.TransformDir(_EqualAreaSquareToSphere((x + 0.5) / n, (y + 0.5) / n)).GetNormalized();
            double t = std::acos(std::clamp(d[1], -1.0, 1.0)) / M_PI;
            double s = 1 - std::fmod(std::atan2(d[0], d[2]) / (2 * M_PI) + 0.5, 1.0);
            dst[size_t(y) * n + x] = src[size_t(std::min(h - 1, int(t * h))) * w + std::min(w - 1, int(s * w))];
        }
    std::string out = TfGetenv("TMPDIR", "/tmp") + "/gonzales_dome_" +
        std::to_string(std::hash<std::string>{}(path + TfStringify(lightToWorld))) + ".exr";
    HioImageSharedPtr image = HioImage::OpenForWriting(out);
    spec.width = spec.height = n;
    spec.data = dst.data();
    return image && image->Write(spec) ? out : std::string();
}

// Everything the prims synced, rebuilt into a Gonzales scene on each render.
class Scene final : public HdRenderParam {
public:
    std::mutex mutex;
    std::map<SdfPath, MeshData> meshes;
    std::map<SdfPath, GfVec3f> materials;
    std::map<SdfPath, LightData> lights;
    std::atomic<int> version{1};

    template <typename F> void Edit(F f) {
        std::lock_guard<std::mutex> lock(mutex);
        f();
        version++;
    }
};

Scene &_Scene(HdRenderParam *param) { return *static_cast<Scene *>(param); }

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
        VtVec3fArray points = delegate->Get(id, HdTokens->points).GetWithDefault<VtVec3fArray>();
        HdMeshUtil util(&topology, id);
        VtVec3iArray triangles;
        VtIntArray primitiveParams;
        util.ComputeTriangleIndices(&triangles, &primitiveParams);
        m.points.assign(points.cdata()->data(), points.cdata()->data() + 3 * points.size());
        m.indices.assign(triangles.cdata()->data(), triangles.cdata()->data() + 3 * triangles.size());
        m.xform = GfMatrix4f(delegate->GetTransform(id));
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

// UsdPreviewSurface's diffuseColor for now.
class Material final : public HdMaterial {
public:
    explicit Material(SdfPath const &id) : HdMaterial(id) {}

    HdDirtyBits GetInitialDirtyBitsMask() const override { return AllDirty; }

    void Sync(HdSceneDelegate *delegate, HdRenderParam *param, HdDirtyBits *dirtyBits) override {
        *dirtyBits = Clean;
        GfVec3f diffuse(0.18f);
        VtValue resource = delegate->GetMaterialResource(GetId());
        if (resource.IsHolding<HdMaterialNetworkMap>()) {
            for (auto const &[terminal, network] : resource.UncheckedGet<HdMaterialNetworkMap>().map)
                for (HdMaterialNode const &node : network.nodes) {
                    if (node.identifier != TfToken("UsdPreviewSurface"))
                        continue;
                    auto it = node.parameters.find(TfToken("diffuseColor"));
                    if (it != node.parameters.end() && it->second.IsHolding<GfVec3f>())
                        diffuse = it->second.UncheckedGet<GfVec3f>();
                }
        }
        _Scene(param).Edit([&] { _Scene(param).materials[GetId()] = diffuse; });
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
    bool IsConverged() const override { return true; }

    // Row 0 of a Hydra buffer is the bottom of the image.
    void Store(unsigned x, unsigned y, float const *value, size_t n) {
        size_t size = HdDataSizeOfFormat(_format);
        _Store(_format, &_buffer[(size_t(_height - 1 - y) * _width + x) * size], value, n);
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
};

void *_Renderer() {
    static void *renderer = gz_renderer_new(TfGetenv("GONZALES_DATA_DIR", GONZALES_DATA_DIR).c_str());
    return renderer;
}

// Renders synchronously whenever the scene, camera or image size changed.
class RenderPass final : public HdRenderPass {
public:
    RenderPass(HdRenderIndex *index, HdRprimCollection const &collection, Scene *scene)
        : HdRenderPass(index, collection), _scene(scene) {}

    bool IsConverged() const override { return true; }

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
        if (!color || !_Renderer())
            return;
        int width = color->GetWidth(), height = color->GetHeight();
        GfMatrix4d view = state->GetWorldToViewMatrix(), proj = state->GetProjectionMatrix();
        int version = _scene->version.load();
        if (version == _version && view == _view && proj == _proj && width == _width && height == _height)
            return;
        _version = version; _view = view; _proj = proj; _width = width; _height = height;

        void *scene = gz_scene_new();
        {
            std::lock_guard<std::mutex> lock(_scene->mutex);
            std::map<SdfPath, int> materials;
            for (auto const &[id, rgb] : _scene->materials)
                materials[id] = gz_scene_add_diffuse(scene, rgb[0], rgb[1], rgb[2]);
            for (auto const &[id, m] : _scene->meshes) {
                if (m.indices.empty())
                    continue;
                auto it = materials.find(m.material);
                int material = it != materials.end() ? it->second
                    : gz_scene_add_diffuse(scene, m.displayColor[0], m.displayColor[1], m.displayColor[2]);
                gz_scene_add_mesh(scene, m.xform.data(), m.points.data(), int(m.points.size() / 3),
                                  m.indices.data(), int(m.indices.size() / 3), nullptr, material);
            }
            bool lit = false;
            for (auto const &[id, l] : _scene->lights) {
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
        }
        // Gonzales cameras look along +z: negate the view's z column.
        GfMatrix4f worldToCamera(view);
        for (int r = 0; r < 4; ++r)
            worldToCamera[r][2] = -worldToCamera[r][2];
        double scale = width >= height ? proj[1][1] : proj[0][0];
        float fov = float(2.0 * std::atan(1.0 / scale) * 180.0 / M_PI);
        gz_scene_set_camera(scene, worldToCamera.data(), fov, width, height);

        HdRenderDelegate *delegate = GetRenderIndex()->GetRenderDelegate();
        int spp = delegate->GetRenderSetting<int>(HdRenderSettingsTokens->convergedSamplesPerPixel, 64);
        int maxDepth = delegate->GetRenderSetting<int>(TfToken("maxDepth"), 5);
        std::vector<float> rgb(size_t(width) * height * 3);
        gz_render(_Renderer(), scene, spp, maxDepth, rgb.data());
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x) {
                float const *p = &rgb[(size_t(y) * width + x) * 3];
                float value[4] = {p[0], p[1], p[2], 1.0f};
                color->Store(x, y, value, 4);
            }
        for (auto &[buffer, clear] : others)
            buffer->Fill(_Floats(clear));
    }

private:
    Scene *_scene;
    int _version = 0, _width = 0, _height = 0;
    GfMatrix4d _view{0.0}, _proj{0.0};
};

class RenderDelegate final : public HdRenderDelegate {
public:
    explicit RenderDelegate(HdRenderSettingsMap const &settings = {})
        : HdRenderDelegate(settings), _registry(std::make_shared<HdResourceRegistry>()) {
        _settingDescriptors = {
            {"Samples per pixel", HdRenderSettingsTokens->convergedSamplesPerPixel, VtValue(64)},
            {"Max depth", TfToken("maxDepth"), VtValue(5)},
        };
        _PopulateDefaultSettings(_settingDescriptors);
    }

    HdRenderParam *GetRenderParam() const override { return &_scene; }

    TfTokenVector const &GetSupportedRprimTypes() const override {
        static TfTokenVector const types = {HdPrimTypeTokens->mesh};
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
        return std::make_shared<RenderPass>(index, collection, &_scene);
    }
    HdInstancer *CreateInstancer(HdSceneDelegate *, SdfPath const &) override { return nullptr; }
    void DestroyInstancer(HdInstancer *instancer) override { delete instancer; }

    HdRprim *CreateRprim(TfToken const &type, SdfPath const &id) override {
        return type == HdPrimTypeTokens->mesh ? new Mesh(id) : nullptr;
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
    void DestroyBprim(HdBprim *prim) override { delete prim; }

    void CommitResources(HdChangeTracker *) override {}

private:
    mutable Scene _scene;
    HdResourceRegistrySharedPtr _registry;
    HdRenderSettingDescriptorList _settingDescriptors;
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
