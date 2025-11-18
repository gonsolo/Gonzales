#ifdef __cplusplus
extern "C" {

#endif
#include <stdbool.h>

// --- Existing Texture System Functions ---
void createTextureSystem();
void destroyTextureSystem();
bool texture(const char *filename_c, float s, float t, float result[3]);

// --- New Tiled Image Writing Functions ---

// Returns an opaque pointer (OIIO::ImageOutput* in C++)
void *openImageForTiledWriting(const char *filename_c, int xres, int yres, int tileWidth, int tileHeight,
                               int channels);

// Writes a single tile using the provided strides.
bool writeSingleTile(void *out, const float *pixels, int xres, int channels, int tx, int ty, int tileWidth,
                     int tileHeight, long long channel_stride, long long x_stride, long long y_stride);

// Safely closes the file and deletes the pointer.
void closeImageOutput(void *out);

#ifdef __cplusplus
}
#endif
