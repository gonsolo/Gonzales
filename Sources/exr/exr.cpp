#include "exr.h"
#include "ImfArray.h"
#include "ImfRgbaFile.h"
#include <vector>

float *readRgbaImpl(const char *fileName, int *width, int *height);
void writeRgbaImpl(const char *fileName, const float *pixels, int width, int height);

#ifdef __cplusplus
extern "C" {
#endif
float *readRgba(const char *fileName, int *width, int *height) {
        return readRgbaImpl(fileName, width, height);
}

void writeRgba(const char *fileName, const float *pixels, const int width, const int height) {
        writeRgbaImpl(fileName, pixels, width, height);
}
#ifdef __cplusplus
}
#endif

float *readRgbaImpl(const char *fileName, int *width, int *height) {
        float *pixels;
#ifdef __x86_64
        Imf::RgbaInputFile file(fileName);
        Imath::Box2i dataWindow = file.dataWindow();
        Imath::V2i min = dataWindow.min;
        Imath::V2i max = dataWindow.max;
        *width = max.x - min.x + 1;
        *height = max.y - min.y + 1;
        Imf::Array2D<Imf::Rgba> array;
        array.resizeErase(*height, *width);
        file.setFrameBuffer(&array[0][0] - min.x - min.y * (*width), 1, *width);
        file.readPixels(min.y, max.y);

        int size = *width * *height * 4;
        pixels = new float[size];
        for (int y = 0; y < *height; ++y) {
                for (int x = 0; x < *width; ++x) {
                        int index = y * (*width) * 4 + x * 4;
                        Imf::Rgba rgba = array[y][x];
                        pixels[index + 0] = rgba.r;
                        pixels[index + 1] = rgba.g;
                        pixels[index + 2] = rgba.b;
                        pixels[index + 3] = rgba.a;
                }
        }
#endif
        return pixels;
}

void writeRgbaImpl(const char *fileName, const float *pixels, int width, int height) {
#ifdef __x86_64
        std::vector<Imf::Rgba> halfPixels(width * height, Imf::Rgba());
        for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                        int index = y * width * 4 + x * 4;
                        Imf::Rgba rgba;
                        rgba.r = pixels[index + 0];
                        rgba.g = pixels[index + 1];
                        rgba.b = pixels[index + 2];
                        rgba.a = pixels[index + 3];
                        halfPixels[y * width + x] = rgba;
                }
        }
        Imf::RgbaOutputFile file(fileName, width, height, Imf::WRITE_RGBA);
        file.setFrameBuffer(halfPixels.data(), 1, width);
        file.writePixels(height);
#endif
}
