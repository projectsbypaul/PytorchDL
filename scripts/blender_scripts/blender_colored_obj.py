import bpy
import random
from pathlib import Path
import pickle

def create_legend(color_label_map, start_location=(0, 0, 0), spacing=20):
    """
    Create a simple legend using colored planes and text.
    
    color_label_map: dict mapping label names or indices to RGBA tuples
    start_location: (x, y, z) start point
    spacing: vertical spacing between legend items
    """
    x, y, z = start_location

    for i, (label, color) in enumerate(color_label_map.items()):
        # Create plane for color
        bpy.ops.mesh.primitive_plane_add(size=10, location=(x, y - i * spacing + 2.5, z ))
        plane = bpy.context.object
        mat = bpy.data.materials.new(name=f"LegendColor_{label}")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        bsdf.inputs['Base Color'].default_value = color
        plane.data.materials.append(mat)

        # Create text
        bpy.ops.object.text_add(location=(x + 10, y - i * spacing, z ))
        text_obj = bpy.context.object
        text_obj.data.body = str(label)
        text_obj.data.size = 10

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
id = "00000002"
obj_path = f"C:/Local_Data/ABC/ABC_parsed_files/{id}/{id}.obj"
obj = load_obj_simple(obj_path)

# Load face colors

# color_map_path = r"C:\src\repos\PytorchDL\data\blender_export\color_map_learned.pkl"
color_map_path = r"C:\src\repos\PytorchDL\data\blender_export\color_map.pkl"

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
    "label_1": (1.0, 0.0, 0.0, 1.0),  # Red
    "label_2": (0.0, 1.0, 0.0, 1.0),  # Green
    "label_3": (0.0, 0.0, 1.0, 1.0),  # Blue
    # Add as needed
}

# Then call:
create_legend(color_label_map, start_location=(100, 0, 0))
