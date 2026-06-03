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
                               int channels, int fullWidth, int fullHeight, int x, int y);

// Writes a single tile using the provided strides.
bool writeSingleTile(void *out, const float *pixels, int xres, int channels, int tx, int ty, int xOffset,
                     int yOffset, int tileWidth, int tileHeight, long long channel_stride, long long x_stride,
                     long long y_stride);

// Safely closes the file and deletes the pointer.
void closeImageOutput(void *out);

// Simple float RGB writer for the Mojo path (step 18).
int write_exr_rgb(const char *filename, const float *rgb, int width, int height,
                  int tile_w, int tile_h);

#ifdef __cplusplus
}
#endif
