#ifdef __cplusplus
extern "C" {

#endif
#include <stdbool.h>

void createTextureSystem();
void destroyTextureSystem();
bool texture(const char *filename_c, float s, float t, float result[3]);
void writeImage(const char *filename_c, const float *pixels, const int xres, const int yres,
                const int tileWidth, const int tileHeight);

#ifdef __cplusplus
}
#endif
