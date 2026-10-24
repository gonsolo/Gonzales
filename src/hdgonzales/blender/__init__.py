# Blender add-on: Gonzales as a Hydra render engine.
import os
import bpy

bl_info = {"name": "Gonzales", "blender": (4, 2, 0), "category": "Render"}

_PLUGIN_DIR = os.environ.get("GONZALES_HYDRA_DIR",
                             os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "..", "build", "hydra"))


class GonzalesRenderEngine(bpy.types.HydraRenderEngine):
    bl_idname = 'GONZALES'
    bl_label = "Gonzales"
    bl_info = "Gonzales GPU path tracer"
    bl_use_preview = False
    bl_use_gpu_context = False
    bl_use_materialx = False
    bl_delegate_id = 'HdGonzalesRendererPlugin'

    def get_render_settings(self, engine_type):
        scene = bpy.context.scene
        result = {'convergedSamplesPerPixel': scene.gonzales_spp, 'maxDepth': scene.gonzales_max_depth}
        if engine_type != 'VIEWPORT':
            result |= {'aovToken:Combined': "color"}
        return result

    def update_render_passes(self, scene, render_layer):
        if render_layer.use_pass_combined:
            self.register_pass(scene, render_layer, 'Combined', 4, 'RGBA', 'COLOR')


def register():
    import pxr.Plug
    pxr.Plug.Registry().RegisterPlugins([os.path.abspath(_PLUGIN_DIR)])
    bpy.types.Scene.gonzales_spp = bpy.props.IntProperty(name="Samples", default=64, min=1)
    bpy.types.Scene.gonzales_max_depth = bpy.props.IntProperty(name="Max Depth", default=12, min=1)
    bpy.utils.register_class(GonzalesRenderEngine)


def unregister():
    bpy.utils.unregister_class(GonzalesRenderEngine)
    del bpy.types.Scene.gonzales_spp
    del bpy.types.Scene.gonzales_max_depth
