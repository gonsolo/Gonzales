#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Viewer* ViewerHandle;

typedef struct {
        float posX, posY, posZ;
        float dirX, dirY, dirZ;
        float upX, upY, upZ;
        int cameraChanged;
} CameraState;

ViewerHandle viewer_create(int width, int height, const char* title);
void viewer_update_framebuffer(ViewerHandle viewer, const float* pixels, int width, int height);
int viewer_should_close(ViewerHandle viewer);
void viewer_poll_events(ViewerHandle viewer);
CameraState viewer_get_camera_state(ViewerHandle viewer);
void viewer_set_camera_state(ViewerHandle viewer, CameraState state);
void viewer_destroy(ViewerHandle viewer);

#ifdef __cplusplus
}
#endif
