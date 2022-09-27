#ifdef __cplusplus
extern "C" {
#endif

void initPtexCache(long int memory);
void initPtexTexture(const char *filename);
void evaluatePtex(const char *filename, long int faceIndex, float u, float v, float rgb[3]);

#ifdef __cplusplus
}
#endif
