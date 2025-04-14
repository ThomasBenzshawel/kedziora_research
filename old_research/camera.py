import os
import numpy as np
from pygltflib import GLTF2
import trimesh
from PIL import Image
import pyrender

os.environ['PYOPENGL_PLATFORM'] = 'egl'

glb_file_path = "/home/benzshawelt/.objaverse/hf-objaverse-v1/glbs/000-011/7a1dddb8703d4ea58a2b3f2ab1b5f2d9.glb"
output_image_path = "output_image.png"

def load_glb(file_path):
    scene = trimesh.load(file_path)
    return scene

def save_image(image_array, output_path):
    image = Image.fromarray(image_array)
    image.save(output_path)

mesh_load = load_glb(glb_file_path)
mesh = mesh_load.to_mesh()

voxelized = mesh.voxelized(pitch=.05)
watertight_mesh = voxelized.marching_cubes

print(f"Is watertight: {watertight_mesh.is_watertight}")

def scale_and_center_mesh(mesh):
    scale = 5 / max(mesh.extents)  # Reduced scale to make object more visible
    mesh.apply_scale(scale)
    center = mesh.bounds.mean(axis=0)
    mesh.apply_translation(-center)
    return mesh

watertight_mesh = scale_and_center_mesh(watertight_mesh)
pics_scene = trimesh.Scene()
pics_scene.add_geometry(watertight_mesh)

def check_scene_contents(scene):
    print(f"Number of geometries in scene: {len(scene.geometry)}")
    for name, geometry in scene.geometry.items():
        print(f"Geometry '{name}':")
        print(f"  Type: {type(geometry)}")
        print(f"  Vertices: {len(geometry.vertices) if hasattr(geometry, 'vertices') else 'N/A'}")
        print(f"  Faces: {len(geometry.faces) if hasattr(geometry, 'faces') else 'N/A'}")
        print(f"  Bounds: {geometry.bounds if hasattr(geometry, 'bounds') else 'N/A'}")

def capture_image(scene, camera_pose, resolution=(1024, 768)):
    check_scene_contents(scene)

    pyrender_scene = pyrender.Scene.from_trimesh_scene(scene)
    
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=resolution[0] / resolution[1])
    camera_node = pyrender_scene.add(camera, pose=camera_pose)
    
    # Add multiple lights
    light_intensity = 2.0
    light_poses = [
        np.eye(4),
        np.array([[ 0, 0, -1, 0],
                  [ 0, 1,  0, 0],
                  [ 1, 0,  0, 0],
                  [ 0, 0,  0, 1]]),
        np.array([[-1, 0,  0, 0],
                  [ 0, 1,  0, 0],
                  [ 0, 0, -1, 0],
                  [ 0, 0,  0, 1]])
    ]
    
    for light_pose in light_poses:
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=light_intensity)
        pyrender_scene.add(light, pose=light_pose)
    
    r = pyrender.OffscreenRenderer(resolution[0], resolution[1])
    color, depth = r.render(pyrender_scene)
    
    if np.any(depth != 0):
        print("Object visible in render")
    else:
        print("No object visible in render")
    
    image = Image.fromarray(color)
    return image

def calculate_camera_pose(direction, scene):
    centroid = scene.centroid
    radius = scene.scale * 2  # Increased camera distance
    
    camera_pos = centroid + np.array(direction) * radius
    
    z = -np.array(direction)
    y = np.array([0.0, 0.0, 1.0])
    x = np.cross(y, z)
    y = np.cross(z, x)
    
    if np.allclose(x, 0) or np.allclose(y, 0) or np.allclose(z, 0):
        print(f"Warning: Zero vector encountered for direction {direction}")
        return None
    
    x = x / np.linalg.norm(x)
    y = y / np.linalg.norm(y)
    z = z / np.linalg.norm(z)
    
    if not (np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(z).all()):
        print(f"Warning: Non-finite values encountered for direction {direction}")
        return None
    
    rotation = np.column_stack([x, y, z])
    
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = camera_pos
    
    return transform

directions = [
    [1, 0, 0], [-1, 0, 0],
    [0, 1, 0], [0, -1, 0],
    [0, 0, 1], [0, 0, -1]
]

if __name__ == "__main__":
    for i, direction in enumerate(directions):
        camera_pose = calculate_camera_pose(direction, pics_scene)
        if camera_pose is not None:
            image = capture_image(pics_scene, camera_pose)
            image.save(f"cube_view_{i}.png")
            print(f"Saved image from direction: {direction}")
        else:
            print(f"Skipping direction: {direction} due to invalid camera pose")

    print("All 6 images saved successfully!")