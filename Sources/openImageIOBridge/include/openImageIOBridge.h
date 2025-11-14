#ifdef __cplusplus
extern "C" {

#endif
#include <stdbool.h>

// --- Existing Texture System Functions ---
void createTextureSystem();
void destroyTextureSystem();
bool texture(const char *filename_c, float s, float t, float result[3]);

// --- New Tiled Image Writing Functions ---

// Returns an opaque pointer (OIIO::ImageOutput*) to the opened file handle.
// Swift assumes ownership and must call closeImageOutput when done.
void* openImageForTiledWriting(const char *filename_c, int xres, int yres,
                               int tileWidth, int tileHeight, int channels);

// Writes the image data body using the file handle.
bool writeTiledImageBody(void *out, const float *pixels, int xres, int yres,
                         int tileWidth, int tileHeight, int channels);

// Safely closes the file and deletes the pointer created by openImageForTiledWriting.
void closeImageOutput(void *out);

// The old void writeImage(...) function is removed.

#ifdef __cplusplus
}
#endif
