# Run by world.py in a separate Blender: renders the world of SOURCE.blend with Cycles into a lat-long EXR in
# Blender's equirectangular convention (u = -atan2(y, x) / 2pi + 0.5, v = elevation / pi + 0.5, row 0 at the top).
#   blender -b --factory-startup --python bake_world.py -- SOURCE.blend OUT.exr WIDTH
import math
import sys

import bpy

source, out, width = sys.argv[-3], sys.argv[-2], int(sys.argv[-1])
with bpy.data.libraries.load(source) as (data_from, data_to):
    data_to.worlds = data_from.worlds
scene = bpy.context.scene
for o in list(scene.objects):
    bpy.data.objects.remove(o)
scene.world = data_to.worlds[0]
camera = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
camera.data.type = 'PANO'
camera.data.panorama_type = 'EQUIRECTANGULAR'
# Looking along +x with +z up puts u = 0.5 at +x, as in Blender's environment textures.
camera.rotation_euler = (math.pi / 2, 0, -math.pi / 2)
scene.collection.objects.link(camera)
scene.camera = camera
scene.render.engine = 'CYCLES'
prefs = bpy.context.preferences.addons['cycles'].preferences
for kind in ('OPTIX', 'CUDA', 'HIP', 'ONEAPI', 'METAL'):
    try:
        prefs.compute_device_type = kind
        prefs.get_devices()
        if any(d.type == kind for d in prefs.devices):
            for d in prefs.devices:
                d.use = d.type == kind
            scene.cycles.device = 'GPU'
            break
    except TypeError:
        continue
scene.cycles.samples = 16
scene.cycles.use_denoising = False
scene.cycles.use_adaptive_sampling = False
scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage = width, width // 2, 100
scene.render.film_transparent = False
scene.view_settings.view_transform = 'Standard'
scene.view_settings.look = 'None'
scene.view_settings.exposure = 0
scene.view_settings.gamma = 1
scene.render.image_settings.file_format = 'OPEN_EXR'
scene.render.image_settings.color_depth = '32'
scene.render.filepath = out
bpy.ops.render.render(write_still=True)
