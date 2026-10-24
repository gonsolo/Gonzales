# Worlds Hydra cannot export (a sky texture, colour nodes, ...) are rendered by Cycles in a separate Blender
# into a lat-long EXR, cached by a hash of the world's nodes, and passed as the "gonzales:world" setting.
import hashlib
import os
import subprocess
import tempfile

import bpy

_BAKER = os.path.join(os.path.dirname(os.path.realpath(__file__)), "bake_world.py")
_WIDTH = 2048


def _simple(world):
    """A world Blender's Hydra export handles: a constant colour or an environment texture."""
    if not world.node_tree:
        return True
    out = next((n for n in world.node_tree.nodes if n.bl_idname == 'ShaderNodeOutputWorld' and n.is_active_output), None)
    if out is None or not out.inputs['Surface'].is_linked:
        return True
    bg = out.inputs['Surface'].links[0].from_node
    if bg.bl_idname != 'ShaderNodeBackground' or bg.inputs['Strength'].is_linked:
        return False
    color = bg.inputs['Color']
    return not color.is_linked or color.links[0].from_node.bl_idname == 'ShaderNodeTexEnvironment'


def _signature(world):
    h = hashlib.sha1()
    for node in sorted(world.node_tree.nodes, key=lambda n: n.name):
        h.update(f"{node.bl_idname} {node.name}".encode())
        for prop in node.bl_rna.properties:
            if prop.type in ('BOOLEAN', 'INT', 'FLOAT', 'STRING', 'ENUM') and not prop.identifier.startswith('bl_'):
                value = getattr(node, prop.identifier, None)
                h.update(f"{prop.identifier}={list(value) if hasattr(value, '__len__') and not isinstance(value, str) else value}".encode())
        image = getattr(node, 'image', None)
        if image:
            h.update(f"{image.filepath} {tuple(image.size)}".encode())
        for s in node.inputs:
            value = getattr(s, 'default_value', None)
            value = list(value) if hasattr(value, '__len__') and not isinstance(value, str) else value
            link = f"{s.links[0].from_node.name}.{s.links[0].from_socket.identifier}" if s.is_linked else ""
            h.update(f"{s.identifier}={value}<{link}".encode())
    return h.hexdigest()[:16]


def texture(world):
    """The path of the baked world, or "" when Hydra's own export suffices."""
    if world is None:
        return ""
    world = world.original
    if _simple(world):
        return ""
    out = os.path.join(tempfile.gettempdir(), f"gonzales_world_{_signature(world)}.exr")
    if not os.path.exists(out):
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "world.blend")
            bpy.data.libraries.write(source, {world}, fake_user=True)
            subprocess.run([bpy.app.binary_path, "-b", "--factory-startup", "--python", _BAKER, "--",
                            source, out, str(_WIDTH)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out if os.path.exists(out) else ""
