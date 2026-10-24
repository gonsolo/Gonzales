// C API of libgonzales.so (src/hdgonzales/capi.mojo).
#pragma once
#ifdef __cplusplus
extern "C" {
#endif
void *gz_renderer_new(const char *data_dir);
void *gz_scene_new(void);
int gz_scene_add_diffuse(void *scene, float r, float g, float b, const char *texture, const char *normal_map);
void gz_scene_add_mesh(void *scene, const float *xform, const float *points, int n_verts, const int *indices,
                       int n_tris, const float *normals, const float *uvs, int material);
void gz_scene_add_area_light(void *scene, const float *xform, const float *points, int n_verts,
                             const int *indices, int n_tris, float r, float g, float b);
void gz_scene_add_dome(void *scene, float r, float g, float b);
void gz_scene_add_dome_map(void *scene, const char *path, float r, float g, float b);
void gz_scene_add_distant(void *scene, float dx, float dy, float dz, float r, float g, float b);
void gz_scene_add_point(void *scene, float x, float y, float z, float r, float g, float b);
void gz_scene_add_sphere_light(void *scene, float x, float y, float z, float radius, float r, float g, float b);
int gz_scene_add_hair(void *scene, float sigma_r, float sigma_g, float sigma_b, float beta_m, float beta_n);
void gz_scene_add_curves(void *scene, const float *control_points, const float *widths, int n_segments, int material);
void gz_scene_set_camera(void *scene, const float *world_to_camera, float fov_deg, int width, int height);
int gz_render(void *renderer, void *scene, int spp, int max_depth, float *out_rgb);
#ifdef __cplusplus
}
#endif
