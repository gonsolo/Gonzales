# Blender add-on: Gonzales as a Hydra render engine.
import os
import bpy

from . import materials, world

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
        spp = scene.gonzales_viewport_spp if engine_type == 'VIEWPORT' else scene.gonzales_spp
        result = {'convergedSamplesPerPixel': spp, 'maxDepth': scene.gonzales_max_depth,
                  'gonzales:materials': materials.overrides(self._materials()),
                  'gonzales:world': world.texture(self._world())}
        if engine_type != 'VIEWPORT':
            result |= {'aovToken:Combined': "color"}
        return result

    # Hydra names materials after the depsgraph's evaluated copies, so remember the depsgraph.
    def update(self, data, depsgraph):
        self._depsgraph = depsgraph
        super().update(data, depsgraph)

    def view_update(self, context, depsgraph):
        self._depsgraph = depsgraph
        super().view_update(context, depsgraph)

    def _world(self):
        depsgraph = getattr(self, '_depsgraph', None)
        return depsgraph.scene.world if depsgraph is not None else bpy.context.scene.world

    def _materials(self):
        depsgraph = getattr(self, '_depsgraph', None)
        if depsgraph is None:
            return [m for m in bpy.data.materials if m.users]
        return [i for i in depsgraph.ids if isinstance(i, bpy.types.Material)]

    def update_render_passes(self, scene, render_layer):
        if render_layer.use_pass_combined:
            self.register_pass(scene, render_layer, 'Combined', 4, 'RGBA', 'COLOR')


class GONZALES_RENDER_PT_sampling(bpy.types.Panel):
    bl_label = "Sampling"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = 'render'

    @classmethod
    def poll(cls, context):
        return context.engine == GonzalesRenderEngine.bl_idname

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        scene = context.scene
        col = layout.column(align=True)
        col.prop(scene, "gonzales_viewport_spp", text="Viewport Samples")
        col.prop(scene, "gonzales_spp", text="Render Samples")
        layout.prop(scene, "gonzales_max_depth")


def register():
    import pxr.Plug
    pxr.Plug.Registry().RegisterPlugins([os.path.abspath(_PLUGIN_DIR)])
    bpy.types.Scene.gonzales_spp = bpy.props.IntProperty(name="Samples", default=64, min=1)
    bpy.types.Scene.gonzales_viewport_spp = bpy.props.IntProperty(name="Viewport Samples", default=32, min=1)
    bpy.types.Scene.gonzales_max_depth = bpy.props.IntProperty(name="Max Depth", default=12, min=1)
    bpy.utils.register_class(GonzalesRenderEngine)
    bpy.utils.register_class(GONZALES_RENDER_PT_sampling)


def unregister():
    bpy.utils.unregister_class(GONZALES_RENDER_PT_sampling)
    bpy.utils.unregister_class(GonzalesRenderEngine)
    del bpy.types.Scene.gonzales_spp
    del bpy.types.Scene.gonzales_viewport_spp
    del bpy.types.Scene.gonzales_max_depth
