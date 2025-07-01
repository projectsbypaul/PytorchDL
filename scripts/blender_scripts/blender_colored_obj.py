import bpy
import pickle
import os
import sys

def create_legend(color_label_map, start_location=(0, 0, 0), spacing_x=40, spacing_y=10, num_columns=3):
    x_start, y_start, z = start_location
    items = list(color_label_map.items())
    num_rows = (len(items) + num_columns - 1) // num_columns

    for i, (label, color) in enumerate(items):
        row = i % num_rows
        col = i // num_rows
        x = x_start + col * spacing_x
        y = y_start - row * spacing_y

        bpy.ops.mesh.primitive_uv_sphere_add(radius=3, location=(x, y + 2.5, z))
        sphere = bpy.context.object
        mat = bpy.data.materials.new(name=f"LegendColor_{label}")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        bsdf.inputs['Base Color'].default_value = color
        sphere.data.materials.append(mat)

        bpy.ops.object.text_add(location=(x + 5, y, z))
        text_obj = bpy.context.object
        text_obj.data.body = str(label)
        text_obj.data.size = 7

def load_obj_simple(filepath):
    verts = []
    faces = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('v '):
                _, x, y, z = line.split()
                verts.append((float(x), float(y), float(z)))
            elif line.startswith('f '):
                parts = line.split()[1:]
                face = [int(p.split('/')[0]) - 1 for p in parts]
                faces.append(face)

    mesh = bpy.data.meshes.new("ImportedMesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new("ImportedObj", mesh)
    bpy.context.collection.objects.link(obj)
    return obj

def color_faces_by_labels(obj, face_colors):
    mesh = obj.data
    mesh.materials.clear()
    color_to_material = {}

    for i, face in enumerate(mesh.polygons):
        color = face_colors[i]
        color_key = tuple(color)
        if color_key not in color_to_material:
            mat = bpy.data.materials.new(name=f"color_{i}")
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links
            nodes.clear()
            output_node = nodes.new(type="ShaderNodeOutputMaterial")
            output_node.location = (300, 0)
            bsdf_node = nodes.new(type="ShaderNodeBsdfPrincipled")
            bsdf_node.location = (0, 0)
            bsdf_node.inputs["Base Color"].default_value = color
            links.new(bsdf_node.outputs["BSDF"], output_node.inputs["Surface"])
            mesh.materials.append(mat)
            color_to_material[color_key] = len(mesh.materials) - 1
        face.material_index = color_to_material[color_key]

# --- SCRIPT ENTRY POINT ---

# Clear scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Get path from CLI argument
paths_file = None
if "--" in sys.argv:
    idx = sys.argv.index("--")
    if idx + 1 < len(sys.argv):
        paths_file = sys.argv[idx + 1]

if not paths_file or not os.path.isfile(paths_file):
    print(f"[ERROR] blender_paths.txt not found or not provided.")
    sys.exit(1)

with open(paths_file, "r") as f:
    lines = [line.strip() for line in f.readlines()]

if len(lines) < 2:
    print(f"[ERROR] blender_paths.txt must contain 2 lines.")
    sys.exit(1)

obj_path, color_map_path = lines

if not os.path.isfile(obj_path):
    print(f"[ERROR] Mesh file not found: {obj_path}")
    sys.exit(1)
if not os.path.isfile(color_map_path):
    print(f"[ERROR] Color map file not found: {color_map_path}")
    sys.exit(1)

print("Loaded mesh:", obj_path)
print("Loaded color map:", color_map_path)

obj = load_obj_simple(obj_path)

with open(color_map_path, "rb") as f:
    face_colors = pickle.load(f)

color_faces_by_labels(obj, face_colors)

# Set to Material Preview
for area in bpy.context.screen.areas:
    if area.type == 'VIEW_3D':
        for space in area.spaces:
            if space.type == 'VIEW_3D':
                space.shading.type = 'MATERIAL'

# Optional legend
color_label_map = {
    'BSpline':   (138, 43, 226, 1.0),
    'Cone':      (0, 0, 255, 1.0),
    'Cylinder':  (255, 0, 0, 1.0),
    'Extrusion': (255, 165, 0, 1.0),
    'Other':     (128, 128, 128, 1.0),
    'Plane':     (255, 20, 147, 1.0),
    'Revolution':(0, 128, 0, 1.0),
    'Sphere':    (128, 0, 0, 1.0),
    'Torus':     (0, 255, 255, 1.0),
    'Void':      (0, 0, 0, 0.0),
}
color_label_map_normalized = {
    label: tuple(c / 255 if i < 3 else c for i, c in enumerate(rgba))
    for label, rgba in color_label_map.items()
}
create_legend(color_label_map_normalized, start_location=(-50, -50, 0))
