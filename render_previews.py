"""
Render preview PNGs of every GridDogs STL, plus a 'family' hero shot of anchors
in use on the 5x5 tile.

Run headless with Blender (no GUI needed):
  blender --background --factory-startup --python render_previews.py
macOS:
  /Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup \
      --python render_previews.py

Reads stl/, writes renders/. Per-part shots show print orientation as exported.
"""

import math
import os

import bpy
from mathutils import Vector

REPO = os.path.dirname(os.path.abspath(__file__))
STL_DIR = os.path.join(REPO, "stl")
OUT_DIR = os.path.join(REPO, "renders")

GREY = (0.42, 0.44, 0.46, 1.0)      # tiles, coupon
ORANGE = (0.80, 0.28, 0.05, 1.0)    # anchors (hero colour)

# Family-shot layout: one representative of every anchor shape.
# Entries are (STL name, z-rotation deg, world x, y, head height).
# x/y chosen so every peg lands on the 14 mm hole lattice of the centred tile.
# The Blender flip (euler X=pi) exactly cancels to_print_orientation's flip, so
# each entry is a plain plan rotation + translation of the IN-USE part: world
# footprint = Rz(rot_z) . in_use + (x, y). All peg positions verified on the
# hole lattice (multiples of 14, within +/-98).
FAMILY = [
    # Tall anchors at the back of the shot.
    ("anchor_round_bumper_40mm.stl", 0, -70, 70, 40.0),
    ("anchor_curve_standard_40mm.stl", 180, -42, 70, 0.0),
    ("anchor_curve_deep_40mm.stl", 180, -14, 70, 0.0),
    ("anchor_curve_bowl_40mm.stl", 180, 14, 70, 0.0),
    # Short anchors in front remain visible.
    ("anchor_wall_diagonal_2x2_20mm.stl", 45, -77, 35, 20.0),
    ("anchor_curve_centre_standard_20mm.stl", 0, -28, 14, 0.0),
    ("anchor_curve_centre_deep_20mm.stl", 0, 0, 14, 0.0),
    ("anchor_bone.stl", 0, 49, 14, 12.0),
    ("anchor_wall_short_20mm.stl", 90, -70, -63, 20.0),
    ("anchor_wall_long_20mm.stl", 0, 0, -70, 20.0),
    ("anchor_wall_diagonal_2x3_20mm.stl", 63.4349488, 63, -70, 20.0),
]


def material(name, rgba, roughness=0.55):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = rgba
    bsdf.inputs["Roughness"].default_value = roughness
    return mat


def import_stl(path, mat):
    bpy.ops.wm.stl_import(filepath=path)
    obj = bpy.context.selected_objects[0]
    obj.data.materials.append(mat)
    try:
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.shade_smooth_by_angle(angle=math.radians(25))
    except AttributeError:
        pass  # older/newer operator name: flat shading still looks fine
    return obj


def fresh_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 64
    scene.cycles.use_denoising = True
    scene.render.image_settings.file_format = "PNG"
    world = bpy.data.worlds.new("world")
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (0.92, 0.92, 0.93, 1.0)
    bg.inputs[1].default_value = 0.55
    scene.world = world
    return scene


def add_ground(size):
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0, 0, 0))
    plane = bpy.context.active_object
    plane.data.materials.append(material("ground", (0.88, 0.88, 0.89, 1.0), 0.9))
    return plane


def add_lights(scale):
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
    sun.data.energy = 3.0
    sun.data.angle = math.radians(8)
    sun.rotation_euler = (math.radians(50), math.radians(-8), math.radians(35))
    bpy.context.collection.objects.link(sun)
    fill = bpy.data.objects.new("fill", bpy.data.lights.new("fill", "AREA"))
    fill.data.energy = 1.5e5 * (scale / 100.0) ** 2
    fill.data.size = scale * 2.5
    fill.location = (-scale * 1.5, -scale * 1.2, scale * 2.0)
    fill.rotation_euler = (math.radians(35), math.radians(-35), 0)
    bpy.context.collection.objects.link(fill)


def add_camera(target, max_dim, res, dist_factor=1.7):
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    cam.data.lens = 50
    cam.data.clip_end = max_dim * 50
    direction = Vector((1.15, -1.35, 1.05)).normalized()
    cam.location = Vector(target) + direction * max_dim * dist_factor
    look = Vector(target) - cam.location
    cam.rotation_euler = look.to_track_quat("-Z", "Y").to_euler()
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    bpy.context.scene.render.resolution_x, bpy.context.scene.render.resolution_y = res


def render_to(name):
    bpy.context.scene.render.filepath = os.path.join(OUT_DIR, name)
    bpy.ops.render.render(write_still=True)
    print("rendered", name)


def bbox(objs):
    bpy.context.view_layer.update()
    pts = [obj.matrix_world @ Vector(c) for obj in objs for c in obj.bound_box]
    lo = Vector(min(p[i] for p in pts) for i in range(3))
    hi = Vector(max(p[i] for p in pts) for i in range(3))
    return lo, hi


def part_render(stl_name):
    fresh_scene()
    mat = material("part", GREY if stl_name.startswith(("tile", "fit", "test")) else ORANGE)
    obj = import_stl(os.path.join(STL_DIR, stl_name), mat)
    lo, hi = bbox([obj])
    dim = hi - lo
    max_dim = max(dim)
    add_ground(max_dim * 60)
    add_lights(max_dim)
    centre = ((lo.x + hi.x) / 2, (lo.y + hi.y) / 2, (lo.z + hi.z) / 2)
    add_camera(centre, max_dim, (1200, 900), dist_factor=2.4)
    render_to(stl_name.replace(".stl", ".png"))


def family_render():
    fresh_scene()
    tile_mat = material("tile", GREY)
    anchor_mat = material("anchor", ORANGE)
    tile_obj = import_stl(os.path.join(STL_DIR, "tile_5x5_gridfinity.stl"), tile_mat)
    objs = [tile_obj]
    plate_top = bbox([tile_obj])[1].z
    for stl_name, rot_z, x, y, head_h in FAMILY:
        obj = import_stl(os.path.join(STL_DIR, stl_name), anchor_mat)
        if stl_name.startswith("anchor_curve_"):
            assert rot_z in (0, 180), "curve placement only handles forward/reverse rotations"
            local_z = [corner[2] for corner in obj.bound_box]
            half_width = (max(local_z) - min(local_z)) / 2
            obj.rotation_euler = (math.pi / 2, 0, math.radians(rot_z))
            obj.location = (x, y + (half_width if rot_z == 0 else -half_width), plate_top)
        else:
            # STLs are in print orientation (head-top at z=0, pegs up); flip about
            # X to put pegs down, then lift so the underside sits on the plate top.
            obj.rotation_euler = (math.pi, 0, math.radians(rot_z))
            obj.location = (x, y, plate_top + head_h)
        objs.append(obj)
    lo, hi = bbox(objs)
    max_dim = max(hi - lo)
    add_ground(max_dim * 60)
    add_lights(max_dim)
    centre = ((lo.x + hi.x) / 2, (lo.y + hi.y) / 2, (lo.z + hi.z) / 2 - max_dim * 0.06)
    add_camera(centre, max_dim, (1600, 1200), dist_factor=1.6)
    render_to("family.png")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    family_render()
    for stl_name in sorted(os.listdir(STL_DIR)):
        if stl_name.endswith(".stl"):
            part_render(stl_name)
