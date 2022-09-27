#ifdef __cplusplus
extern "C" {
#endif

float *readRgba(const char *fileName, int *width, int *height);
void writeRgba(const char *fileName, const float *pixels, const int width, const int height);

#ifdef __cplusplus
}
#endif
