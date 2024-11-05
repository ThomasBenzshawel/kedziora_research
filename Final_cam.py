import trimesh

import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import pyrender
import numpy as np
from PIL import Image
import cv2


def get_camera_poses(scene):
    centroid = scene.centroid
    bounds = scene.bounds
    #todo
    diagonal = np.linalg.norm(bounds[1] - bounds[0])
    
    # Define camera positions relative to the centroid
    positions = [
        [0, 0, diagonal],  # Front
        [0, 0, -diagonal],  # Back
        [diagonal, 0, 0],  # Right
        [-diagonal, 0, 0],  # Left
        [0, diagonal, 0.1*diagonal],  # Up (slightly offset)
        [0, -diagonal, 0.1*diagonal],  # Down (slightly offset)
    ]
    
    camera_poses = []
    for pos in positions:
        pose = np.eye(4)
        pose[:3, 3] = centroid + np.array(pos)
        pose[:3, :3] = look_at(pose[:3, 3], centroid)
        camera_poses.append(pose)
    
    return camera_poses, diagonal

def look_at(eye, target):
    #potential bug
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    
    to_side = np.cross(forward, np.array([0, 1, 0]))
    if np.allclose(to_side, 0):
        to_side = np.cross(forward, np.array([1, 0, 0]))
    
    to_side = to_side / np.linalg.norm(to_side)
    up = np.cross(to_side, forward)
    
    rotation_matrix = np.column_stack((to_side, up, -forward))
    return rotation_matrix


def render_glb(glb_path, output_prefix, resolution=(800, 600)):
    scene = trimesh.load(glb_path)

    pyrender_scene = pyrender.Scene.from_trimesh_scene(scene)

    camera_poses, scene_diagonal = get_camera_poses(scene)
    fov = 2 * np.arctan(scene_diagonal / (2 * scene_diagonal))
    camera = pyrender.PerspectiveCamera(yfov=fov)


    for pos in camera_poses:
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.75)
        pyrender_scene.add(light, pose=pos)


    r = pyrender.OffscreenRenderer(viewport_width=resolution[0],
                                viewport_height=resolution[1])

    perspectives = ['front', 'back', 'right', 'left', 'up', 'down']
    for i, pose in enumerate(camera_poses):
        camera_node = pyrender_scene.add(camera, pose=pose)
        
        # Initialize light with low intensity
        light_intensity = 1.5
        light = pyrender.SpotLight(color=np.ones(3), intensity=light_intensity,
                                   innerConeAngle=np.pi/16.0,
                                   outerConeAngle=np.pi/6.0)
        light_node = pyrender_scene.add(light, pose=pose)

        for iteration in range(10):
            color, _ = r.render(pyrender_scene)
            
            if not is_black_and_white(color):
                break
            
            if iteration < 9:
                # Increase light intensity and try again
                light_intensity *= 5
                pyrender_scene.remove_node(light_node)
                light = pyrender.SpotLight(color=np.ones(3), intensity=light_intensity,
                                           innerConeAngle=np.pi/16.0,
                                           outerConeAngle=np.pi/6.0)
                light_node = pyrender_scene.add(light, pose=pose)
            else:
                # If we've reached 10 iterations, use the original (black and white) image
                print(f"Warning: {perspectives[i]} view remained black and white after 10 iterations.")

        img = Image.fromarray(color)
        output_path = f"{output_prefix}_{perspectives[i]}.jpg"  # Changed file extension to .jpg
        img.save(output_path, 'JPEG', quality=95)  # Specify JPEG format and quality
        # print(f"Image saved to {output_path}")

        pyrender_scene.remove_node(camera_node)
        pyrender_scene.remove_node(light_node)

def is_black_and_white(image, color_threshold=10):
    # Convert to HSV for better color analysis
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Check the standard deviation of the Hue channel
    hue_std = np.std(hsv[:,:,0])
    
    # Check the number of unique colors
    unique_colors = np.unique(image.reshape(-1, 3), axis=0)
    
    # Consider it color if there's significant hue variation or many unique colors
    return hue_std < 5 and len(unique_colors) <= color_threshold




# Process all files in the Objaverse dataset
import json
import os
from tqdm import tqdm  # For progress bar


def process_objaverse_files(json_path, output_dir):
    # Read and parse the gzipped JSON file
    with open(json_path, 'rt', encoding='utf-8') as f:
        object_paths = json.load(f)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Iterate through the object paths
    for uid, file_path in tqdm(object_paths.items(), desc="Processing files"):
        # Construct the full path to the GLB file
        glb_file = os.path.join(os.path.dirname(json_path), file_path)
        
        # Check if the file exists
        if not os.path.exists(glb_file):
            # print(f"File not found: {glb_file}")
            continue

        #make a directory for each object
        os.makedirs(os.path.join(output_dir, uid), exist_ok=True)
        
        # Create an output prefix for this object 
        output_prefix = os.path.join(output_dir, uid, uid)
        
        try:
            # Render the GLB file
            render_glb(glb_file, output_prefix)
        except Exception as e:
            print(f"Error processing {glb_file}: {str(e)}")

# Example usage
json_path = "/home/benzshawelt/.objaverse/hf-objaverse-v1/object-paths.json"
output_dir = "/home/benzshawelt/Research/objaverse_images"
process_objaverse_files(json_path, output_dir)


        
# # Example usage
# # glb_file = "/home/benzshawelt/.objaverse/hf-objaverse-v1/glbs/000-011/7a1dddb8703d4ea58a2b3f2ab1b5f2d9.glb"
# # glb_file = "/home/benzshawelt/.objaverse/hf-objaverse-v1/glbs/000-038/a72cbed60b2b434e8da695fd389fde5b.glb"
# # glb_file = "/home/benzshawelt/.objaverse/hf-objaverse-v1/glbs/000-049/a86575a96a2e4c63bfa4a0ade255f19c.glb"
# glb_file = "/home/benzshawelt/.objaverse/hf-objaverse-v1/glbs/000-111/3f030f580d2c4cdf8096b4a3a63cecaf.glb"
# output_prefix = "output_image_3"
# render_glb(glb_file, output_prefix)
