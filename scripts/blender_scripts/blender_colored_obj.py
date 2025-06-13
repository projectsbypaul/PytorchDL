import bpy
import random
from pathlib import Path
import pickle

def create_legend(color_label_map, start_location=(0, 0, 0), spacing_x=40, spacing_y=10, num_columns=3):
    """
    Create a legend using colored spheres and text, arranged in columns with custom spacing.

    color_label_map: dict mapping label names or indices to RGBA tuples
    start_location: (x, y, z) start point
    spacing_x: horizontal spacing between columns
    spacing_y: vertical spacing between rows
    num_columns: number of columns in the legend
    """
    x_start, y_start, z = start_location
    items = list(color_label_map.items())
    num_rows = (len(items) + num_columns - 1) // num_columns  # ceil division

    for i, (label, color) in enumerate(items):
        row = i % num_rows
        col = i // num_rows

        x = x_start + col * spacing_x
        y = y_start - row * spacing_y

        # Create sphere
        bpy.ops.mesh.primitive_uv_sphere_add(radius=3, location=(x, y + 2.5, z))
        sphere = bpy.context.object
        mat = bpy.data.materials.new(name=f"LegendColor_{label}")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        bsdf.inputs['Base Color'].default_value = color
        sphere.data.materials.append(mat)

        # Create text
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
                face = [int(p.split('/')[0]) - 1 for p in parts]  # OBJ is 1-indexed
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
        color = face_colors[i]  # Now color is RGBA tuple

        color_key = tuple(color)  # Make sure it's hashable
        if color_key not in color_to_material:
            # Create material
            mat = bpy.data.materials.new(name=f"color_{i}")
            mat.use_nodes = True

            # Get node tree
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links

            # Clear existing nodes
            nodes.clear()

            # Create nodes
            output_node = nodes.new(type="ShaderNodeOutputMaterial")
            output_node.location = (300, 0)

            bsdf_node = nodes.new(type="ShaderNodeBsdfPrincipled")
            bsdf_node.location = (0, 0)

            # Assign specific color
            bsdf_node.inputs["Base Color"].default_value = color

            # Connect BSDF to Output
            links.new(bsdf_node.outputs["BSDF"], output_node.inputs["Surface"])

            # Add material to object
            mesh.materials.append(mat)
            color_to_material[color_key] = len(mesh.materials) - 1

        # Assign material to face
        face.material_index = color_to_material[color_key]
# --- SCRIPT ENTRY POINT ---

# Clear all objects
for obj in bpy.data.objects:
    bpy.data.objects.remove(obj, do_unlink=True)
# Load mesh
test_id = "test_2"
obj_path = r"H:\ABC_Demo\source" + "\\" + test_id + "\\" + test_id + ".obj"
print(obj_path)
obj = load_obj_simple(obj_path)

# Load face colors

color_map_path = r"H:\ABC_Demo\blender" + "\\" + test_id + "_color_map.pkl"
#color_map_path = r"C:\src\repos\PytorchDL\data\blender_export\color_map.pkl"

# Load the color map
with open(color_map_path, "rb") as f:
    face_colors = pickle.load(f)

# Load face colors
with open(color_map_path, "rb") as f:
    face_colors = pickle.load(f)
# Assign color by labels
color_faces_by_labels(obj, face_colors)

# Optional: switch to material preview mode
for area in bpy.context.screen.areas:
    if area.type == 'VIEW_3D':
        for space in area.spaces:
            if space.type == 'VIEW_3D':
                space.shading.type = 'MATERIAL'
                

# Map colors to labels (you can define your own)
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
    # Add as needed
}

color_label_map_normalized = {
    label: tuple(c / 255 if i < 3 else c for i, c in enumerate(rgba))
    for label, rgba in color_label_map.items()
}

# Then call:
create_legend(color_label_map_normalized, start_location=(-50, -50, 0))
