#ifdef __cplusplus
extern "C" {
#endif

void createTextureSystem();
void destroyTextureSystem();
bool texture(const char* filename_c, float s, float t, float result[3]);

#ifdef __cplusplus
}
#endif

