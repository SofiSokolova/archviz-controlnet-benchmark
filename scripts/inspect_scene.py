"""Print a JSON summary of an open .blend file.

Usage:
    blender -b --factory-startup --disable-autoexec scene.blend --python scripts/inspect_scene.py
"""
import json
import math
import os

import bpy


def camera_info(obj):
    return {
        "name": obj.name,
        "location": [round(v, 3) for v in obj.matrix_world.translation],
        "rotation_deg": [round(math.degrees(v), 1) for v in obj.matrix_world.to_euler()],
        "lens_mm": round(obj.data.lens, 1),
        "sensor_mm": round(obj.data.sensor_width, 1),
        "constraints": [c.type for c in obj.constraints],
        "animated": obj.animation_data is not None,
    }


def main():
    scene = bpy.context.scene
    missing = [
        img.name
        for img in bpy.data.images
        if img.source == "FILE"
        and not img.packed_file
        and not os.path.exists(bpy.path.abspath(img.filepath, library=img.library))
    ]
    summary = {
        "blender_version": bpy.app.version_string,
        "file_version": ".".join(str(v) for v in bpy.data.version),
        "scene": scene.name,
        "scenes": [s.name for s in bpy.data.scenes],
        "engine": scene.render.engine,
        "resolution": [scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage],
        "samples": scene.cycles.samples,
        "view_transform": scene.view_settings.view_transform,
        "look": scene.view_settings.look,
        "view_layers": [vl.name for vl in scene.view_layers if vl.use],
        "frame": [scene.frame_start, scene.frame_current, scene.frame_end],
        "active_camera": scene.camera.name if scene.camera else None,
        "cameras": [camera_info(o) for o in scene.objects if o.type == "CAMERA"],
        "libraries": len(bpy.data.libraries),
        "objects": len(scene.objects),
        "meshes": sum(1 for o in scene.objects if o.type == "MESH"),
        "materials": len(bpy.data.materials),
        "images": len(bpy.data.images),
        "images_missing": missing[:20],
        "images_missing_count": len(missing),
    }
    print("INSPECT_JSON " + json.dumps(summary, ensure_ascii=False))


main()
