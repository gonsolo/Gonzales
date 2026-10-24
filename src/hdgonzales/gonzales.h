// C API of libgonzales.so (src/hdgonzales/capi.mojo).
#pragma once
#ifdef __cplusplus
extern "C" {
#endif
void *gz_renderer_new(const char *data_dir);
void *gz_scene_new(void);
int gz_scene_add_diffuse(void *scene, float r, float g, float b, const char *texture, const char *normal_map);
int gz_scene_add_coated(void *scene, float r, float g, float b, const char *texture, const char *normal_map, float alpha,
                        float ior);
void gz_scene_add_mesh(void *scene, const float *xform, const float *points, int n_verts, const int *indices,
                       int n_tris, const float *normals, const float *uvs, int material);
void gz_scene_add_area_light(void *scene, const float *xform, const float *points, int n_verts,
                             const int *indices, int n_tris, float r, float g, float b);
void gz_scene_add_dome(void *scene, float r, float g, float b);
void gz_scene_add_dome_map(void *scene, const char *path, float r, float g, float b);
void gz_scene_add_distant(void *scene, float dx, float dy, float dz, float r, float g, float b);
void gz_scene_add_point(void *scene, float x, float y, float z, float r, float g, float b);
void gz_scene_add_sphere_light(void *scene, float x, float y, float z, float radius, float r, float g, float b);
int gz_scene_add_hair(void *scene, float sigma_r, float sigma_g, float sigma_b, float beta_m, float beta_n, float ior, float alpha);
int gz_scene_add_dielectric(void *scene, float ior, float alpha);
void gz_scene_add_curves(void *scene, const float *control_points, const float *widths, int n_segments, int material,
                         const int *materials);
void gz_scene_set_camera(void *scene, const float *world_to_camera, float fov_deg, int width, int height);
void gz_scene_set_lens(void *scene, float radius, float focal_distance);
void *gz_session_new(void *renderer, void *scene, int spp, int max_depth);
int gz_session_set_camera(void *session, const float *world_to_camera, float fov_deg, int width, int height);
void gz_session_set_samples(void *session, int spp, int max_depth);
void gz_session_set_lens(void *session, float radius, float focal_distance);
int gz_session_render(void *session, int n_spp);
int gz_session_resolve(void *session, float *out_rgb, int denoise);
void gz_session_free(void *session);
#ifdef __cplusplus
}
#endif
