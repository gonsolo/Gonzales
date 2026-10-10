#include <OpenImageIO/imageio.h>
#include <OpenImageIO/imagecache.h>
#include <OpenImageIO/texture.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

std::shared_ptr<OIIO::TextureSystem> textureSystem;

// sRGB → linear (same conversion as CPU shader's _srgb_to_linear)
static inline float srgb_to_linear(float c) {
        return c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f);
}

// Built through OIIO's own uint8→float conversion so results are bit-identical
// to reading the image as FLOAT and converting every texel.
static std::array<float, 256> make_uint8_lut(bool decode) {
        std::array<unsigned char, 256> in;
        for (int i = 0; i < 256; ++i)
                in[i] = static_cast<unsigned char>(i);
        std::array<float, 256> lut;
        OIIO::convert_pixel_values(OIIO::TypeDesc::UINT8, in.data(), OIIO::TypeDesc::FLOAT, lut.data(), 256);
        if (decode)
                for (float &v : lut)
                        v = srgb_to_linear(v);
        return lut;
}

// raw != 0 keeps linear data (e.g. normal maps); only sRGB colour textures get decoded.
static bool wants_srgb_decode(const char *filename, int raw) {
        const bool hdr = strstr(filename, ".exr") != nullptr || strstr(filename, ".hdr") != nullptr ||
                         strstr(filename, ".pfm") != nullptr;
        return !hdr && !raw;
}

template <typename T, typename Map>
static void to_linear_rgb(const T *src, int64_t n, int nc, float *dst, Map map) {
        for (int64_t i = 0; i < n; ++i) {
                const T *p = src + i * nc;
                const float r = nc > 0 ? map(p[0]) : 0.0f;
                dst[i * 3 + 0] = r;
                dst[i * 3 + 1] = nc > 1 ? map(p[1]) : r;
                dst[i * 3 + 2] = nc > 2 ? map(p[2]) : r;
        }
}

// ── Embedded working-colour-space (chromaticities) → renderer's Rec.709/sRGB
// primaries ────────────────────────────────────────────────────────────────
// Some HDRI environment maps in the corpus (bistro/sanmiguel/villa/sportscar/
// etc.'s "sky.exr") carry an OpenEXR "chromaticities" attribute tagging them
// as ACES2065-1 (AP0) data -- R(0.7347,0.2653) G(0,1) B(0.0001,-0.077)
// White~(0.3217,0.3377), NOT the renderer's native Rec.709/sRGB primaries
// (D65 white 0.3127,0.3290). Reading those floats as if they were already
// Rec.709 (this loader's prior behaviour) desaturates and hue-shifts the
// whole image -- AP0's much wider red primary reads as excess red, turning
// a saturated blue sky visibly purple/lavender and inflating luminance by
// ~10% even before any scene-specific compounding. Confirmed against a real
// pbrt-v4 render of bistro's sky.exr alone (no geometry): gonzales's R
// channel averaged 47% brighter than pbrt's, G/B within 4% -- exactly the
// asymmetric-per-channel signature of an unconverted wide-gamut primaries
// mismatch, not a uniform exposure/scale error (which affects all channels
// equally) or a texture-mapping/rotation bug (which would show as spatial
// displacement, not a per-channel colour cast at a fixed pixel).
//
// Converts any embedded primaries+white to XYZ (standard chromaticity->XYZ
// construction), Bradford-adapts between the two white points, and composes
// the full source-RGB -> XYZ -> dest-RGB matrix. Verified independently
// (scratch Python) to reproduce the widely-published ACES AP0 -> linear
// Rec.709 (Bradford D60->D65) matrix to 5 decimal places, so this is not a
// special case for ACES specifically -- any tagged primaries get the same
// treatment, matching how pbrt-v4 itself handles embedded EXR chromaticities.

static void invert3x3(const double m[9], double out[9]) {
        double a = m[0], b = m[1], c = m[2];
        double d = m[3], e = m[4], f = m[5];
        double g = m[6], h = m[7], i = m[8];
        double A =  (e * i - f * h);
        double B = -(d * i - f * g);
        double C =  (d * h - e * g);
        double D = -(b * i - c * h);
        double E =  (a * i - c * g);
        double F = -(a * h - b * g);
        double G =  (b * f - c * e);
        double H = -(a * f - c * d);
        double I =  (a * e - b * d);
        double det = a * A + b * B + c * C;
        if (std::fabs(det) < 1e-12) {
                // Degenerate primaries -- fall back to identity rather than divide by ~0.
                out[0]=1; out[1]=0; out[2]=0;
                out[3]=0; out[4]=1; out[5]=0;
                out[6]=0; out[7]=0; out[8]=1;
                return;
        }
        double invDet = 1.0 / det;
        out[0] = A * invDet; out[1] = D * invDet; out[2] = G * invDet;
        out[3] = B * invDet; out[4] = E * invDet; out[5] = H * invDet;
        out[6] = C * invDet; out[7] = F * invDet; out[8] = I * invDet;
}

static void mat3_mul(const double a[9], const double b[9], double out[9]) {
        for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c)
                        out[r * 3 + c] = a[r * 3 + 0] * b[0 * 3 + c] + a[r * 3 + 1] * b[1 * 3 + c] +
                                         a[r * 3 + 2] * b[2 * 3 + c];
}

static void mat3_vec(const double m[9], const double v[3], double out[3]) {
        out[0] = m[0] * v[0] + m[1] * v[1] + m[2] * v[2];
        out[1] = m[3] * v[0] + m[4] * v[1] + m[5] * v[2];
        out[2] = m[6] * v[0] + m[7] * v[1] + m[8] * v[2];
}

static void chromaticity_to_XYZ(double x, double y, double xyz[3]) {
        if (y == 0.0) { xyz[0] = xyz[1] = xyz[2] = 0.0; return; }
        xyz[1] = 1.0;
        xyz[0] = x / y;
        xyz[2] = (1.0 - x - y) / y;
}

// RGB(primaries,white) -> XYZ, columns = R/G/B chromaticities scaled so the
// white point maps to XYZ with Y=1 (the standard primaries-matrix method).
static void primaries_to_XYZ(double rx, double ry, double gx, double gy, double bx, double by, double wx,
                             double wy, double out[9]) {
        double Xr[3], Xg[3], Xb[3], Xw[3];
        chromaticity_to_XYZ(rx, ry, Xr);
        chromaticity_to_XYZ(gx, gy, Xg);
        chromaticity_to_XYZ(bx, by, Xb);
        chromaticity_to_XYZ(wx, wy, Xw);
        double P[9] = {Xr[0], Xg[0], Xb[0], Xr[1], Xg[1], Xb[1], Xr[2], Xg[2], Xb[2]};
        double Pinv[9];
        invert3x3(P, Pinv);
        double S[3];
        mat3_vec(Pinv, Xw, S);
        out[0] = Xr[0] * S[0]; out[1] = Xg[0] * S[1]; out[2] = Xb[0] * S[2];
        out[3] = Xr[1] * S[0]; out[4] = Xg[1] * S[1]; out[5] = Xb[1] * S[2];
        out[6] = Xr[2] * S[0]; out[7] = Xg[2] * S[1]; out[8] = Xb[2] * S[2];
}

// Bradford chromatic-adaptation matrix (XYZ_src-white -> XYZ_dst-white).
static void bradford_adapt(const double Xw_src[3], const double Xw_dst[3], double out[9]) {
        double B[9] = {0.8951, 0.2664, -0.1614, -0.7502, 1.7135, 0.0367, 0.0389, -0.0685, 1.0296};
        double Binv[9];
        invert3x3(B, Binv);
        double cone_src[3], cone_dst[3];
        mat3_vec(B, Xw_src, cone_src);
        mat3_vec(B, Xw_dst, cone_dst);
        double diag[9] = {0};
        diag[0] = cone_dst[0] / cone_src[0];
        diag[4] = cone_dst[1] / cone_src[1];
        diag[8] = cone_dst[2] / cone_src[2];
        double tmp[9];
        mat3_mul(diag, B, tmp);
        mat3_mul(Binv, tmp, out);
}

// Renderer's working space: Rec.709/sRGB primaries, D65 white.
static const double REC709_R[2] = {0.64, 0.33};
static const double REC709_G[2] = {0.30, 0.60};
static const double REC709_B[2] = {0.15, 0.06};
static const double REC709_W[2] = {0.3127, 0.3290};

// Fills `m` (row-major 3x3, applied as m*rgb) with the source-primaries ->
// Rec.709/sRGB conversion and returns true, unless the embedded primaries
// already match Rec.709/D65 closely (common case -- most textures aren't
// tagged, or are already native), in which case returns false and leaves
// `m` untouched so callers can skip the per-pixel matrix multiply entirely.
static bool chromaticities_to_working_space_matrix(const float chroma[8], double m[9]) {
        double rx = chroma[0], ry = chroma[1], gx = chroma[2], gy = chroma[3];
        double bx = chroma[4], by = chroma[5], wx = chroma[6], wy = chroma[7];
        static const double TOL = 0.002;
        if (std::fabs(rx - REC709_R[0]) < TOL && std::fabs(ry - REC709_R[1]) < TOL &&
            std::fabs(gx - REC709_G[0]) < TOL && std::fabs(gy - REC709_G[1]) < TOL &&
            std::fabs(bx - REC709_B[0]) < TOL && std::fabs(by - REC709_B[1]) < TOL &&
            std::fabs(wx - REC709_W[0]) < TOL && std::fabs(wy - REC709_W[1]) < TOL)
                return false;
        double M_src_to_XYZ[9];
        primaries_to_XYZ(rx, ry, gx, gy, bx, by, wx, wy, M_src_to_XYZ);
        double M_dst_to_XYZ[9];
        primaries_to_XYZ(REC709_R[0], REC709_R[1], REC709_G[0], REC709_G[1], REC709_B[0], REC709_B[1],
                         REC709_W[0], REC709_W[1], M_dst_to_XYZ);
        double M_XYZ_to_dst[9];
        invert3x3(M_dst_to_XYZ, M_XYZ_to_dst);
        double Xw_src[3], Xw_dst[3];
        chromaticity_to_XYZ(wx, wy, Xw_src);
        chromaticity_to_XYZ(REC709_W[0], REC709_W[1], Xw_dst);
        double M_cat[9];
        bradford_adapt(Xw_src, Xw_dst, M_cat);
        double tmp[9];
        mat3_mul(M_cat, M_src_to_XYZ, tmp);
        mat3_mul(M_XYZ_to_dst, tmp, m);
        return true;
}

// Reads the EXR "chromaticities" attribute (redX,redY,greenX,greenY,blueX,
// blueY,whiteX,whiteY) if present; returns false (leaving chroma untouched)
// when the file has none -- the overwhelmingly common case.
static bool read_chromaticities(const OIIO::ImageSpec &spec, float chroma[8]) {
        return spec.getattribute("chromaticities", OIIO::TypeDesc(OIIO::TypeDesc::FLOAT, 8), chroma);
}

static void apply_working_space_matrix(float *data, int64_t n, const double m[9]) {
        for (int64_t i = 0; i < n; ++i) {
                double r = data[i * 3 + 0], g = data[i * 3 + 1], b = data[i * 3 + 2];
                data[i * 3 + 0] = float(m[0] * r + m[1] * g + m[2] * b);
                data[i * 3 + 1] = float(m[3] * r + m[4] * g + m[5] * b);
                data[i * 3 + 2] = float(m[6] * r + m[7] * g + m[8] * b);
        }
}

// --- Exposed C functions (Must be compiled with C linkage) ---
#ifdef __cplusplus
extern "C" {
#endif

// Returns an OIIO::ImageOutput* (void* in Swift)
OIIO::ImageOutput *openImageForTiledWriting(const char *filename_c, int xres, int yres, int tileWidth,
                                            int tileHeight, int channels, int fullWidth, int fullHeight,
                                            int x, int y) {

        std::unique_ptr<OIIO::ImageOutput> out_uptr = OIIO::ImageOutput::create(filename_c);

        if (!out_uptr)
                return nullptr;

        OIIO::ImageSpec spec(xres, yres, channels, OIIO::TypeDesc::FLOAT);
        spec.tile_width = tileWidth;
        spec.tile_height = tileHeight;
        spec.full_x = 0;
        spec.full_y = 0;
        spec.full_width = fullWidth;
        spec.full_height = fullHeight;
        spec.x = x;
        spec.y = y;

        if (!out_uptr->open(filename_c, spec)) {
                std::cerr << "ERROR: Could not open file: " << out_uptr->geterror() << std::endl;
                return nullptr;
        }

        return out_uptr.release();
}

// Function to write a single tile (called inside the Swift loop)
bool writeSingleTile(OIIO::ImageOutput *out, const float *pixels, int xres, int channels, int tx, int ty,
                     int xOffset, int yOffset, int tileWidth, int tileHeight, ptrdiff_t channel_stride,
                     ptrdiff_t x_stride, ptrdiff_t y_stride) {

        int buffer_x = tx * tileWidth;
        int buffer_y = ty * tileHeight;

        ptrdiff_t index_offset = (ptrdiff_t)buffer_y * xres + buffer_x;
        const float *tile_ptr = pixels + index_offset * channels;

        OIIO::image_span<const float> tile_span(tile_ptr, (ptrdiff_t)channels, (size_t)tileWidth,
                                                (size_t)tileHeight, 1, channel_stride, x_stride, y_stride);

        int target_x = buffer_x + xOffset;
        int target_y = buffer_y + yOffset;

        if (!out->write_tile(target_x, target_y, 0, tile_span)) {
                std::cerr << "ERROR: Failed to write tile (" << target_x << "," << target_y
                          << "): " << out->geterror() << std::endl;
                return false;
        }
        return true;
}

void closeImageOutput(OIIO::ImageOutput *out) {
        if (out) {
                out->close();
                delete out;
        }
}

// Existing texture system functions (kept for completeness)
void createTextureSystem() { textureSystem = OIIO::TextureSystem::create(); }

void destroyTextureSystem() { OIIO::TextureSystem::destroy(textureSystem); }

// PFM stores its rows bottom-to-top (the format's origin is the lower-left
// corner), and pbrt-v4's ReadPFM flips them accordingly (util/image.cpp).
// OIIO's PFM reader does NOT (its row 0 is the file's first row, i.e. the
// bottom of the image) even though its PFM writer does -- so every .pfm
// texture was read upside-down relative to pbrt. Invisible for any content
// symmetric under a vertical flip (a constant sky, a radial gradient, the
// equal-area map's own sky/ground mask), which is why flat-colour A/B tests
// all matched; the 13 Bitterli scenes lit by a .pfm env map were not.
static bool is_pfm(const char *filename) {
        const char *dot = strrchr(filename, '.');
        return dot && (strcmp(dot, ".pfm") == 0 || strcmp(dot, ".PFM") == 0);
}

bool texture(const char *filename_c, float s, float t, float result[3]) {
        if (is_pfm(filename_c)) t = 1.0f - t;
        OIIO::ustring filename(filename_c);
        OIIO::TextureOpt options;
        float dsdx = 0;
        float dtdx = 0;
        float dsdy = 0;
        float dtdy = 0;
        int nchannels = 3;
        bool ok = textureSystem->texture(filename, options, s, t, dsdx, dtdx, dsdy, dtdy, nchannels, result);
        // TextureOpt::fill defaults to 0, so a single-channel (greyscale) or
        // two-channel source image sampled here for 3 channels comes back
        // with the missing channel(s) at 0 instead of replicated luminance --
        // e.g. a grey texture turns pure red (R=lum, G=B=0). Detect and
        // replicate, matching load_texture_rgb's already-correct nc-aware
        // handling used by the GPU upload path.
        if (ok) {
                const OIIO::ImageSpec *spec = textureSystem->imagecache()->imagespec(filename);
                if (spec) {
                        int nc = spec->nchannels;
                        if (nc <= 1) {
                                result[1] = result[0];
                                result[2] = result[0];
                        } else if (nc == 2) {
                                result[2] = result[0];
                        }
                }
        }
        return ok;
}

int load_texture_rgb(const char *filename, float **data, int *width, int *height, int raw) {
        auto in = OIIO::ImageInput::open(filename);
        if (!in) return 0;
        const OIIO::ImageSpec &spec = in->spec();
        *width  = spec.width;
        *height = spec.height;
        const int64_t n = int64_t(spec.width) * spec.height;
        const int nc = spec.nchannels;
        const bool decode = wants_srgb_decode(filename, raw);
        *data = (float *)malloc(n * 3 * sizeof(float));
        if (!*data) return 0;
        bool ok;
        if (spec.format == OIIO::TypeDesc::UINT8 && spec.channelformats.empty()) {
                static const std::array<float, 256> lut_linear = make_uint8_lut(false);
                static const std::array<float, 256> lut_srgb = make_uint8_lut(true);
                const std::array<float, 256> &lut = decode ? lut_srgb : lut_linear;
                std::unique_ptr<unsigned char[]> buf(new unsigned char[n * nc]);
                ok = in->read_image(0, 0, 0, nc, OIIO::TypeDesc::UINT8, buf.get());
                if (ok)
                        to_linear_rgb(buf.get(), n, nc, *data, [&](unsigned char v) { return lut[v]; });
        } else {
                std::unique_ptr<float[]> buf(new float[n * nc]);
                ok = in->read_image(0, 0, 0, nc, OIIO::TypeDesc::FLOAT, buf.get());
                if (ok && is_pfm(filename)) {
                        const int64_t row = int64_t(spec.width) * nc;
                        for (int y = 0; y < spec.height / 2; ++y)
                                std::swap_ranges(buf.get() + y * row, buf.get() + (y + 1) * row,
                                                 buf.get() + (spec.height - 1 - y) * row);
                }
                if (ok)
                        to_linear_rgb(buf.get(), n, nc, *data, [&](float v) { return decode ? srgb_to_linear(v) : v; });
        }
        in->close();
        // The buffer is uninitialized now, so a failed read must not be handed back as a texture.
        if (!ok) {
                free(*data);
                *data = nullptr;
                return 0;
        }
        // Embedded working-colour-space conversion (see the comment above
        // chromaticities_to_working_space_matrix) -- e.g. the corpus's ACES
        // AP0-tagged "sky.exr" HDRIs, read as-is until now.
        float chroma[8];
        if (read_chromaticities(spec, chroma)) {
                double m[9];
                if (chromaticities_to_working_space_matrix(chroma, m))
                        apply_working_space_matrix(*data, n, m);
        }
        return 1;
}

int free_texture_rgb(float *data) {
        free(data);
        return 0;
}

// decode: 0 = linear, 1 = sRGB, 2 = gamma 2.2 (pbrt's default Ptex encoding)
void texture_uint8_lut(int decode, float *out) {
        if (decode == 2) {
                for (int i = 0; i < 256; ++i)
                        out[i] = std::pow(i / 255.0f, 2.2f);
                return;
        }
        const std::array<float, 256> lut = make_uint8_lut(decode != 0);
        std::copy(lut.begin(), lut.end(), out);
}

int load_texture_u8(const char *filename, int raw, unsigned char **data, int *width, int *height,
                    int *channels, int *srgb) {
        auto in = OIIO::ImageInput::open(filename);
        if (!in) return 0;
        const OIIO::ImageSpec &spec = in->spec();
        if (spec.format != OIIO::TypeDesc::UINT8 || !spec.channelformats.empty() || spec.nchannels < 1) {
                in->close();
                return 0;
        }
        const int w = spec.width, h = spec.height, nc = spec.nchannels;
        const int64_t n = int64_t(w) * h;
        auto *buf = static_cast<unsigned char *>(malloc(n * nc));
        if (!buf || !in->read_image(0, 0, 0, nc, OIIO::TypeDesc::UINT8, buf)) {
                free(buf);
                in->close();
                return 0;
        }
        in->close();
        if (nc == 1 || nc == 3) {
                *data = buf;
        } else {
                // Same mapping as load_texture_rgb: G from channel 1 (alpha for 2-channel
                // sources), B from channel 2 or copied from R.
                auto *rgb = static_cast<unsigned char *>(malloc(n * 3));
                if (!rgb) {
                        free(buf);
                        return 0;
                }
                for (int64_t i = 0; i < n; ++i) {
                        const unsigned char *p = buf + i * nc;
                        rgb[i * 3 + 0] = p[0];
                        rgb[i * 3 + 1] = p[1];
                        rgb[i * 3 + 2] = nc > 2 ? p[2] : p[0];
                }
                free(buf);
                *data = rgb;
        }
        *width = w;
        *height = h;
        *channels = nc == 1 ? 1 : 3;
        *srgb = wants_srgb_decode(filename, raw) ? 1 : 0;
        return 1;
}

// pbrt-v4 GPU's single-channel ("float") image texture, which is what a
// `Shape "texture alpha"` reads (textures.cpp, the lumTextureCache path): an
// RGBA image whose alpha is not all ones contributes its A channel, anything
// else the average of R, G, B; bytes are taken raw (cudaReadModeNormalizedFloat,
// no sRGB decode). pbrt's CPU build differs -- it reads channel 0 (red) of the
// sRGB-decoded image -- and the GPU rule is the one that treats a leaf PNG's
// alpha channel as its cut-out, so it is the one followed. malloc'd, one byte
// per texel -- free with free_texture_u8.
int load_alpha_mask(const char *filename, unsigned char **data, int *width, int *height) {
        auto in = OIIO::ImageInput::open(filename);
        if (!in) return 0;
        const OIIO::ImageSpec &spec = in->spec();
        const int w = spec.width, h = spec.height, nc = spec.nchannels;
        if (nc < 1) {
                in->close();
                return 0;
        }
        const int64_t n = int64_t(w) * h;
        std::vector<unsigned char> buf(n * nc);
        if (!in->read_image(0, 0, 0, nc, OIIO::TypeDesc::UINT8, buf.data())) {
                in->close();
                return 0;
        }
        in->close();
        auto *mask = static_cast<unsigned char *>(malloc(n));
        if (!mask) return 0;
        bool use_alpha = false;
        if (nc >= 4)
                for (int64_t i = 0; i < n && !use_alpha; ++i)
                        use_alpha = buf[i * nc + 3] != 255;
        for (int64_t i = 0; i < n; ++i) {
                const unsigned char *p = buf.data() + i * nc;
                if (use_alpha)
                        mask[i] = p[3];
                else if (nc >= 3)
                        mask[i] = (unsigned char)((int(p[0]) + int(p[1]) + int(p[2]) + 1) / 3);
                else
                        mask[i] = p[0];
        }
        *data = mask;
        *width = w;
        *height = h;
        return 1;
}

int free_texture_u8(unsigned char *data) {
        free(data);
        return 0;
}

static bool is_hdr_ext(const char *filename) {
        const char *dot = strrchr(filename, '.');
        if (!dot) return true;
        return strcmp(dot, ".exr") == 0 || strcmp(dot, ".hdr") == 0 || strcmp(dot, ".pfm") == 0;
}

struct RGB {
        float r, g, b;

        static RGB from(const float *p) { return {p[0], p[1], p[2]}; }

        float luminance() const { return 0.2126f * r + 0.7152f * g + 0.0722f * b; }

        RGB reinhard() const {
                float lum = luminance();
                float scale = lum > 1e-6f ? (lum / (1.0f + lum)) / lum : 1.0f;
                return {r * scale, g * scale, b * scale};
        }

        RGB to_srgb() const { return {linear_to_srgb(r), linear_to_srgb(g), linear_to_srgb(b)}; }

        uint8_t to_u8(float ch) const { return (uint8_t)std::min(255.0f, ch * 255.0f + 0.5f); }
        void store_u8(uint8_t *p) const { p[0] = to_u8(r); p[1] = to_u8(g); p[2] = to_u8(b); }

private:
        static float linear_to_srgb(float x) {
                if (x <= 0.0f) return 0.0f;
                if (x >= 1.0f) return 1.0f;
                return x <= 0.0031308f ? 12.92f * x : 1.055f * std::pow(x, 1.0f / 2.4f) - 0.055f;
        }
};

// Write a float RGB buffer.  EXR/HDR → float32 tiles.
// PNG/JPG/etc. → Reinhard-on-luminance tonemap + sRGB gamma → uint8.
// Returns 1 on success, 0 on failure.
int write_image_rgb(const char *filename, const float *rgb, int width, int height,
                    int tile_w, int tile_h) {
        auto out = OIIO::ImageOutput::create(filename);
        if (!out) {
                std::cerr << "write_image_rgb: cannot create writer for " << filename << std::endl;
                return 0;
        }
        if (is_hdr_ext(filename)) {
                OIIO::ImageSpec spec(width, height, 3, OIIO::TypeDesc::FLOAT);
                spec.tile_width = tile_w;
                spec.tile_height = tile_h;
                if (!out->open(filename, spec)) return 0;
                out->write_image(OIIO::TypeDesc::FLOAT, rgb);
        } else {
                int n = width * height;
                std::vector<uint8_t> ldr(n * 3);
                for (int i = 0; i < n; ++i)
                        RGB::from(rgb + i * 3).reinhard().to_srgb().store_u8(ldr.data() + i * 3);
                OIIO::ImageSpec spec(width, height, 3, OIIO::TypeDesc::UINT8);
                if (!out->open(filename, spec)) return 0;
                out->write_image(OIIO::TypeDesc::UINT8, ldr.data());
        }
        out->close();
        return 1;
}

// See oiio.h for the dataWindow/displayWindow contract.
int write_image_rgb_windowed(const char *filename, const float *rgb, int width, int height,
                             int full_width, int full_height, int x, int y,
                             int tile_w, int tile_h) {
        auto out = OIIO::ImageOutput::create(filename);
        if (!out) {
                std::cerr << "write_image_rgb_windowed: cannot create writer for " << filename << std::endl;
                return 0;
        }
        if (is_hdr_ext(filename)) {
                OIIO::ImageSpec spec(width, height, 3, OIIO::TypeDesc::FLOAT);
                spec.tile_width = tile_w;
                spec.tile_height = tile_h;
                spec.full_x = 0;
                spec.full_y = 0;
                spec.full_width = full_width;
                spec.full_height = full_height;
                spec.x = x;
                spec.y = y;
                if (!out->open(filename, spec)) return 0;
                out->write_image(OIIO::TypeDesc::FLOAT, rgb);
        } else {
                int n = width * height;
                std::vector<uint8_t> ldr(n * 3);
                for (int i = 0; i < n; ++i)
                        RGB::from(rgb + i * 3).reinhard().to_srgb().store_u8(ldr.data() + i * 3);
                OIIO::ImageSpec spec(width, height, 3, OIIO::TypeDesc::UINT8);
                if (!out->open(filename, spec)) return 0;
                out->write_image(OIIO::TypeDesc::UINT8, ldr.data());
        }
        out->close();
        return 1;
}

// n interleaved float channels with the given comma-separated names (e.g.
// "R,G,B,Albedo.R,Albedo.G,Albedo.B,N.X,N.Y,N.Z,Z", pbrt-v4 GBufferFilm's
// naming), tagged with the same dataWindow/displayWindow convention as
// write_image_rgb_windowed. Float formats only: an LDR target (PNG, ...)
// cannot hold arbitrary channels, so it gets the first three as a
// tone-mapped RGB image, like write_image_rgb.
int write_image_channels(const char *filename, const float *data, int width, int height,
                         int nchannels, const char *channel_names,
                         int full_width, int full_height, int x, int y,
                         int tile_w, int tile_h) {
        auto out = OIIO::ImageOutput::create(filename);
        if (!out) {
                std::cerr << "write_image_channels: cannot create writer for " << filename << std::endl;
                return 0;
        }
        if (!is_hdr_ext(filename)) {
                int n = width * height;
                std::vector<uint8_t> ldr(n * 3);
                for (int i = 0; i < n; ++i) {
                        float rgb[3] = {data[i * nchannels], data[i * nchannels + (nchannels > 1 ? 1 : 0)],
                                        data[i * nchannels + (nchannels > 2 ? 2 : 0)]};
                        RGB::from(rgb).reinhard().to_srgb().store_u8(ldr.data() + i * 3);
                }
                OIIO::ImageSpec spec(width, height, 3, OIIO::TypeDesc::UINT8);
                if (!out->open(filename, spec)) return 0;
                out->write_image(OIIO::TypeDesc::UINT8, ldr.data());
                out->close();
                return 1;
        }
        OIIO::ImageSpec spec(width, height, nchannels, OIIO::TypeDesc::FLOAT);
        std::vector<std::string> names;
        std::string all(channel_names ? channel_names : "");
        size_t start = 0;
        while (start <= all.size()) {
                size_t comma = all.find(',', start);
                if (comma == std::string::npos) comma = all.size();
                names.push_back(all.substr(start, comma - start));
                start = comma + 1;
        }
        if ((int)names.size() == nchannels)
                spec.channelnames = names;
        spec.tile_width = tile_w;
        spec.tile_height = tile_h;
        spec.full_x = 0;
        spec.full_y = 0;
        spec.full_width = full_width;
        spec.full_height = full_height;
        spec.x = x;
        spec.y = y;
        if (!out->open(filename, spec)) return 0;
        out->write_image(OIIO::TypeDesc::FLOAT, data);
        out->close();
        return 1;
}

#ifdef __cplusplus
}
#endif

// Mean colour of a Ptex file: each sampled face's 1x1 reduction, decoded with `gamma`
// (pbrt's ptex "encoding" default is gamma 2.2) and averaged in linear space.
#include <Ptexture.h>
#include <map>
#include <mutex>
#include <string>
// Per-face linear mean colours of a Ptex file, computed once per (file, gamma): each face is read at
// <= 8x8 and decoded per texel (decoding a 1x1 average darkens contrasty faces; full res is far too slow).
static const std::vector<float> *ptex_file_faces(const char *filename, float gamma) {
        static std::mutex mu;
        static std::map<std::pair<std::string, float>, std::vector<float>> cache;
        std::lock_guard<std::mutex> lock(mu);
        auto key = std::make_pair(std::string(filename), gamma);
        auto it = cache.find(key);
        if (it != cache.end())
                return it->second.empty() ? nullptr : &it->second;
        std::vector<float> &out = cache[key];
        Ptex::String err;
        PtexPtr<PtexTexture> tex(PtexTexture::open(filename, err));
        if (!tex)
                return nullptr;
        int nfaces = tex->numFaces(), nch = tex->numChannels();
        if (nfaces <= 0 || nch <= 0)
                return nullptr;
        out.assign(3 * static_cast<size_t>(nfaces), 0.0f);
        std::vector<char> raw(Ptex::DataSize(tex->dataType()) * nch);
        std::vector<float> px(nch);
        for (int f = 0; f < nfaces; ++f) {
                Ptex::Res full = tex->getFaceInfo(f).res;
                Ptex::Res r(std::min<int>(full.ulog2, 3), std::min<int>(full.vlog2, 3));
                PtexPtr<PtexFaceData> fd(tex->getData(f, r));
                if (!fd)
                        continue;
                int w = r.u(), h = r.v();
                double m[3] = {0, 0, 0};
                for (int v = 0; v < h; ++v)
                        for (int u = 0; u < w; ++u) {
                                fd->getPixel(u, v, raw.data());
                                Ptex::ConvertToFloat(px.data(), raw.data(), tex->dataType(), nch);
                                for (int c = 0; c < 3; ++c)
                                        m[c] += std::pow(std::max(px[nch >= 3 ? c : 0], 0.0f), gamma);
                        }
                for (int c = 0; c < 3; ++c)
                        out[3 * f + c] = static_cast<float>(m[c] / (w * h));
        }
        return &out;
}
extern "C" int ptex_average_rgb(const char *filename, float gamma, float out[3]) {
        const std::vector<float> *faces = ptex_file_faces(filename, gamma);
        if (!faces)
                return 0;
        double sum[3] = {0, 0, 0};
        size_t n = faces->size() / 3;
        for (size_t f = 0; f < n; ++f)
                for (int c = 0; c < 3; ++c)
                        sum[c] += (*faces)[3 * f + c];
        for (int c = 0; c < 3; ++c)
                out[c] = static_cast<float>(sum[c] / n);
        return 1;
}

// Per-face mean colour of a Ptex file; malloc'd nfaces*3 floats, release with free_texture_rgb.
extern "C" int ptex_face_rgb(const char *filename, float gamma, float **data, int *nfaces_out) {
        const std::vector<float> *faces = ptex_file_faces(filename, gamma);
        if (!faces)
                return 0;
        float *out = static_cast<float *>(std::malloc(sizeof(float) * faces->size()));
        std::memcpy(out, faces->data(), sizeof(float) * faces->size());
        *data = out;
        *nfaces_out = static_cast<int>(faces->size() / 3);
        return 1;
}

// ── Packed Ptex for the renderer's texture table ─────────────────────────────
// Blob = nfaces * 8 header bytes, then RGB8 texels (gamma-2.2 encoded).
// Header entry: uint32 texel offset of the face's level 0, uint8 log2 width, uint8 log2 height,
// uint8 wanted log2 (written by the renderer; 255 = nothing finer exists), uint8 flags
// (1 = the offset is into the shared page pool instead of this file's own texels, 2 = used).
// Each face stores its full mip chain, levels halving on both axes (clamped at 1) down to 1x1.
static std::mutex ptex_blob_mu;
static std::map<std::string, std::vector<unsigned char>> ptex_blobs;

static int ptex_planned_log2 = 4;       // uniform cap that fits the memory budget
static int ptex_base_log2 = 0;          // resident cap when paging
static long long ptex_pool_budget = 0;  // page pool bytes when paging
static bool ptex_paging_on = false;

static bool ptex_paging() { return ptex_paging_on && ptex_pool_budget > 0 && !std::getenv("GONZALES_PTEX_LOG2RES"); }

static int ptex_max_log2() {
        const char *e = std::getenv("GONZALES_PTEX_LOG2RES");
        int v = e ? std::atoi(e) : (ptex_paging() ? ptex_base_log2 : ptex_planned_log2);
        return std::min(std::max(v, 0), 12);
}

extern "C" void ptex_set_plan(int cap, int base, long long pool_bytes) {
        ptex_planned_log2 = cap;
        ptex_base_log2 = base;
        ptex_pool_budget = pool_bytes;
}
extern "C" void ptex_set_paging(int on) { ptex_paging_on = on != 0; }

// Every level carries a one-texel border copied from the neighbouring faces, so a bilinear
// lookup blends across face edges: a 2^ul x 2^vl level takes (2^ul + 2) * (2^vl + 2) texels.
static size_t ptex_chain_texels(int ul, int vl) {
        size_t n = 0;
        for (;;) {
                n += ((size_t(1) << ul) + 2) * ((size_t(1) << vl) + 2);
                if (ul == 0 && vl == 0)
                        return n;
                ul = std::max(ul - 1, 0);
                vl = std::max(vl - 1, 0);
        }
}

struct PtexTexelReader {
        PtexTexture *tex;
        int nch;
        bool is_u8;
        std::vector<char> raw;
        std::vector<float> px;
        explicit PtexTexelReader(PtexTexture *t)
            : tex(t), nch(t->numChannels()), is_u8(t->dataType() == Ptex::dt_uint8),
              raw(Ptex::DataSize(t->dataType()) * t->numChannels()), px(t->numChannels()) {}
        // RGB8, gamma 2.2 for files that are not 8-bit.
        void get(PtexFaceData *fd, int u, int v, unsigned char *dst) {
                fd->getPixel(u, v, raw.data());
                if (is_u8) {
                        const unsigned char *r = reinterpret_cast<const unsigned char *>(raw.data());
                        for (int c = 0; c < 3; ++c)
                                dst[c] = r[nch >= 3 ? c : 0];
                        return;
                }
                Ptex::ConvertToFloat(px.data(), raw.data(), tex->dataType(), nch);
                for (int c = 0; c < 3; ++c) {
                        float lin = std::min(std::max(px[nch >= 3 ? c : 0], 0.0f), 1.0f);
                        dst[c] = static_cast<unsigned char>(std::pow(lin, 1.0f / 2.2f) * 255.0f + 0.5f);
                }
        }
};

// Write face f's mip chain from (ul, vl) down to 1x1, each level with its border, to dst.
// Edges run counter-clockwise (bottom, right, top, left), so position k along one face's edge
// meets position n-1-k along its neighbour's.
static void ptex_read_chain(PtexTexture *tex, int f, int ul, int vl, unsigned char *dst) {
        PtexTexelReader rd(tex);
        const Ptex::FaceInfo &info = tex->getFaceInfo(f);
        for (;;) {
                const int w = 1 << ul, h = 1 << vl, stride = w + 2;
                auto at = [&](int x, int y) { return dst + (static_cast<size_t>(y + 1) * stride + (x + 1)) * 3; };
                PtexPtr<PtexFaceData> fd(tex->getData(f, Ptex::Res(ul, vl)));
                for (int y = 0; y < h; ++y)
                        for (int x = 0; x < w; ++x) {
                                if (fd)
                                        rd.get(fd, x, y, at(x, y));
                                else
                                        at(x, y)[0] = at(x, y)[1] = at(x, y)[2] = 0;
                        }
                for (int e = 0; e < 4; ++e) {
                        const int n = (e & 1) ? h : w, nl = (e & 1) ? vl : ul;
                        const int af = info.adjface(e), ae = info.adjedge(e);
                        PtexPtr<PtexFaceData> nd;
                        int aw = 1, ah = 1, al = 0;
                        if (af >= 0 && af < tex->numFaces()) {
                                const Ptex::FaceInfo &ai = tex->getFaceInfo(af);
                                if (ai.isSubface() == info.isSubface()) {
                                        const int fa = (ae & 1) ? ai.res.vlog2 : ai.res.ulog2, fp = (ae & 1) ? ai.res.ulog2 : ai.res.vlog2;
                                        const int delta = std::max(fa - nl, 0);
                                        al = fa - delta;
                                        const int pl = std::max(fp - delta, 0);
                                        const Ptex::Res ares = (ae & 1) ? Ptex::Res(pl, al) : Ptex::Res(al, pl);
                                        nd.reset(tex->getData(af, ares));
                                        aw = 1 << ares.ulog2;
                                        ah = 1 << ares.vlog2;
                                }
                        }
                        for (int k = 0; k < n; ++k) {
                                const int x = e == 0 ? k : e == 1 ? w : e == 2 ? w - 1 - k : -1;
                                const int y = e == 0 ? -1 : e == 1 ? k : e == 2 ? h : h - 1 - k;
                                if (!nd) {   // mesh border: repeat the face's own edge texel
                                        std::memcpy(at(x, y), at(std::min(std::max(x, 0), w - 1), std::min(std::max(y, 0), h - 1)), 3);
                                        continue;
                                }
                                const int j = ((n - 1 - k) << al) >> nl;
                                const int au = ae == 0 ? j : ae == 1 ? aw - 1 : ae == 2 ? aw - 1 - j : 0;
                                const int av = ae == 0 ? 0 : ae == 1 ? j : ae == 2 ? ah - 1 : ah - 1 - j;
                                rd.get(nd, au, av, at(x, y));
                        }
                }
                const int cx[4] = {-1, w, w, -1}, cy[4] = {-1, -1, h, h};
                for (int c = 0; c < 4; ++c) {   // corners: mean of the two border texels beside them
                        const unsigned char *p = at(cx[c] < 0 ? 0 : w - 1, cy[c]), *q = at(cx[c], cy[c] < 0 ? 0 : h - 1);
                        for (int i = 0; i < 3; ++i)
                                at(cx[c], cy[c])[i] = static_cast<unsigned char>((p[i] + q[i] + 1) / 2);
                }
                dst += static_cast<size_t>(w + 2) * (h + 2) * 3;
                if (ul == 0 && vl == 0)
                        break;
                ul = std::max(ul - 1, 0);
                vl = std::max(vl - 1, 0);
        }
}

// bytes[c] += packed size of this file with faces capped at 2^c per side, c = 0..12.
extern "C" int ptex_bytes_by_cap(const char *filename, long long *bytes) {
        Ptex::String err;
        PtexPtr<PtexTexture> tex(PtexTexture::open(filename, err));
        if (!tex)
                return 0;
        const int nfaces = tex->numFaces();
        for (int f = 0; f < nfaces; ++f) {
                Ptex::Res full = tex->getFaceInfo(f).res;
                for (int c = 0; c <= 12; ++c)
                        bytes[c] += ptex_chain_texels(std::min<int>(full.ulog2, c), std::min<int>(full.vlog2, c)) * 3 + 8;
        }
        return 1;
}

// Demand paging lives in ptex_cache.mojo; these are its file accessors.
extern "C" long long ptex_pool_size() { return ptex_paging() ? ptex_pool_budget : 0; }
extern "C" void *ptex_open(const char *filename) {
        Ptex::String err;
        return PtexTexture::open(filename, err);
}
extern "C" void ptex_close(void *tex) { static_cast<PtexTexture *>(tex)->release(); }
extern "C" void ptex_face_res(void *tex, int f, int *res) {
        Ptex::Res full = static_cast<PtexTexture *>(tex)->getFaceInfo(f).res;
        res[0] = full.ulog2;
        res[1] = full.vlog2;
}
extern "C" void ptex_read_face(void *tex, int f, int ul, int vl, unsigned char *dst) {
        ptex_read_chain(static_cast<PtexTexture *>(tex), f, ul, vl, dst);
}

// Pack `filename`; *paged_id is 0 when its faces can be paged in at finer resolutions, else -1.
extern "C" int ptex_packed_size(const char *filename, long long *nbytes, int *nfaces_out, int *paged_id) {
        Ptex::String err;
        PtexPtr<PtexTexture> tex(PtexTexture::open(filename, err));
        if (!tex)
                return 0;
        const int nfaces = tex->numFaces(), nch = tex->numChannels();
        if (nfaces <= 0 || nch <= 0)
                return 0;
        const int cap = ptex_max_log2();
        const bool paged = ptex_paging();
        std::vector<unsigned char> blob(static_cast<size_t>(nfaces) * 8);
        size_t texels = 0;
        for (int f = 0; f < nfaces; ++f) {
                Ptex::Res full = tex->getFaceInfo(f).res;
                const int ul = std::min<int>(full.ulog2, cap), vl = std::min<int>(full.vlog2, cap);
                const uint32_t off = static_cast<uint32_t>(texels);
                unsigned char *e = &blob[static_cast<size_t>(f) * 8];
                std::memcpy(e, &off, 4);
                e[4] = static_cast<unsigned char>(ul);
                e[5] = static_cast<unsigned char>(vl);
                e[6] = (paged && (ul < full.ulog2 || vl < full.vlog2)) ? 0 : 255;
                e[7] = 0;
                texels += ptex_chain_texels(ul, vl);
        }
        blob.resize(static_cast<size_t>(nfaces) * 8 + texels * 3);
        for (int f = 0; f < nfaces; ++f) {
                const unsigned char *e = &blob[static_cast<size_t>(f) * 8];
                uint32_t off;
                std::memcpy(&off, e, 4);
                ptex_read_chain(tex, f, e[4], e[5], &blob[static_cast<size_t>(nfaces) * 8 + static_cast<size_t>(off) * 3]);
        }
        *nbytes = static_cast<long long>(blob.size());
        *nfaces_out = nfaces;
        *paged_id = -1;
        if (paged)
                *paged_id = 0;
        std::lock_guard<std::mutex> lock(ptex_blob_mu);
        ptex_blobs[filename] = std::move(blob);
        return 1;
}

// Copy the blob built by ptex_packed_size into `dst` and drop it.
extern "C" int ptex_packed_take(const char *filename, unsigned char *dst) {
        std::lock_guard<std::mutex> lock(ptex_blob_mu);
        auto it = ptex_blobs.find(filename);
        if (it == ptex_blobs.end())
                return 0;
        std::memcpy(dst, it->second.data(), it->second.size());
        ptex_blobs.erase(it);
        return 1;
}
