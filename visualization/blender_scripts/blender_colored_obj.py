import bpy
import random
import pickle

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
id = "00000377"
obj_path = f"C:/Local_Data/ABC/ABC_parsed_files/{id}/{id}.obj"
obj = load_obj_simple(obj_path)

# Load face colors
with open(r"C:\src\repos\PytorchDL\data\blender_export\color_map.pkl", "rb") as f:
    face_colors = pickle.load(f)
    
# Assign color by labels
color_faces_by_labels(obj, face_colors)

# Optional: switch to material preview mode
for area in bpy.context.screen.areas:
    if area.type == 'VIEW_3D':
        for space in area.spaces:
            if space.type == 'VIEW_3D':
                space.shading.type = 'MATERIAL'
