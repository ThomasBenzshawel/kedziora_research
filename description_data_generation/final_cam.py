import trimesh

import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import pyrender
import numpy as np
from PIL import Image
import cv2
import json
import os
from tqdm import tqdm  # For progress bar
import time

VERIFICATION = "./check_objaverse_images/"

def get_camera_poses(scene):
    centroid = scene.centroid
    bounds = scene.bounds
    diagonal = np.linalg.norm(bounds[1] - bounds[0])
    
    # Define camera positions relative to the centroid
    positions = [
        [0, 0, diagonal],  # Front
        [0, 0, -diagonal],  # Back
        [diagonal, 0, 0],  # Right
        [-diagonal, 0, 0],  # Left
        [0, diagonal, 0.075*diagonal],  # Up (slightly offset)
        [0, -diagonal, 0.075*diagonal],  # Down (slightly offset)
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

    pyrender_scene = pyrender.Scene.from_trimesh_scene(scene, bg_color=[0.6, 0.6, 0.6], ambient_light = [1, 1, 1])
    
    camera_poses, scene_diagonal = get_camera_poses(scene)
    fov = 2 * np.arctan(scene_diagonal / (2 * scene_diagonal))
    camera = pyrender.PerspectiveCamera(yfov=fov)

    for pos in camera_poses:
        light = pyrender.DirectionalLight(color=[0.6, 0.6, 0.6], intensity=2)
        pyrender_scene.add(light, pose=pos)

    perspectives = ['front', 'back', 'right', 'left', 'up', 'down']
    for i, pose in enumerate(camera_poses):
        light_intensity = 2  # Initial light intensity
        light = pyrender.SpotLight(color=np.ones(3), intensity=light_intensity,
                                            innerConeAngle=np.pi/8.0,
                                            outerConeAngle=np.pi/4.0)
        light_node = pyrender_scene.add(light, pose=pose)


        for iteration in range(10):

            camera_node = pyrender_scene.add(camera, pose=pose)
            r = pyrender.OffscreenRenderer(viewport_width=resolution[0],
        viewport_height=resolution[1], point_size=1.0)

            color, _ = r.render(pyrender_scene)
            r.delete()
            is_bw = is_black_and_white(color)

            pyrender_scene.remove_node(camera_node)

            
            if is_bw == "color":
                # Save the image if it's not black and white
                break
            
            if iteration < 9:
                if is_bw == "black":
                    print(f"Image is black, increasing light intensity for {perspectives[i]} view.")
                    # Increase light intensity and try again
                    light_intensity *= 1.75
                    pyrender_scene.remove_node(light_node)
                    light = pyrender.SpotLight(color=np.ones(3), intensity=light_intensity,
                                            innerConeAngle=np.pi/8.0,
                                            outerConeAngle=np.pi/4.0)
                    light_node = pyrender_scene.add(light, pose=pose)
                elif is_bw == "white":
                    print(f"Image is white, decreasing light intensity for {perspectives[i]} view.")
                    # Decrease light intensity and try again
                    light_intensity *= 0.5
                    pyrender_scene.remove_node(light_node)
                    light = pyrender.SpotLight(color=np.ones(3), intensity=light_intensity,
                                            innerConeAngle=np.pi/8.0,
                                            outerConeAngle=np.pi/4.0)
                    light_node = pyrender_scene.add(light, pose=pose)
            else:
                # If we've reached 10 iterations, use the original (black and white) image
                print(f"Warning: {perspectives[i]} view remained black and white after 10 iterations.")
                img = Image.fromarray(color)
                output_path = f"{VERIFICATION}{os.path.basename(glb_path).split('.')[0]}_{perspectives[i]}.jpg"
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                # get the name of the object
                # Save the image with the object name
                img.save(output_path, 'JPEG', quality=95)  # Specify JPEG format and quality

        img = Image.fromarray(color)
        output_path = f"{output_prefix}_{perspectives[i]}.jpg"  #
        img.save(output_path, 'JPEG', quality=95)  # Specify JPEG format and quality
        # print(f"Image saved to {output_path}")

        pyrender_scene.remove_node(light_node)

def is_black_and_white(image, color_threshold=10, saturation_threshold=20, value_range_threshold=30):
    # Convert to HSV for better color analysis
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Check the standard deviation and mean of the Hue channel
    hue_std = np.std(hsv[:,:,0])

    # Analyze saturation channel - crucial for detecting colorfulness
    sat_mean = np.mean(hsv[:,:,1])
    sat_std = np.std(hsv[:,:,1])

    # Analyze value (brightness) channel to distinguish grayscale from white/black
    val_mean = np.mean(hsv[:,:,2])
    val_range = np.max(hsv[:,:,2]) - np.min(hsv[:,:,2])

    # Calculate histogram of hue and look for peaks
    hist_hue = cv2.calcHist([hsv], [0], None, [36], [0, 180])
    hist_hue = hist_hue / np.sum(hist_hue)  # Normalize
    hue_peaks = np.sum(hist_hue > 0.05)  # Count significant hue peaks

    # Check the number of unique colors
    unique_colors = np.unique(image.reshape(-1, 3), axis=0)

    # Comprehensive assessment of whether the image is black and white
    is_bw = (
        (hue_std < 5) and                    # Low hue variation
        (sat_mean < saturation_threshold) and # Low overall saturation
        (sat_std < saturation_threshold/2) and # Low saturation variation
        (hue_peaks <= 2) and                  # Few dominant hues
        (len(unique_colors) <= color_threshold or 
         (val_range < value_range_threshold and sat_mean < 10))  # Few colors or low dynamic range with low saturation
    )

    if is_bw:
        # Check if it's predominantly dark or light
        if val_mean < 50:
            return "black"
        else:
            return "white"
    else:
        return "color"

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
            # print(f"Object not found: {glb_file}")
            continue

        #make a directory for each object if the directory does not exist, if it does, skip that object
        if os.path.exists(os.path.join(output_dir, uid)):
            print(f"Directory already exists for {uid}, skipping.")
            continue
        else:
            os.makedirs(os.path.join(output_dir, uid), exist_ok=True)
        
        # Create an output prefix for this object 
        output_prefix = os.path.join(output_dir, uid, uid)
        
        try:
            # Render the GLB file
            render_glb(glb_file, output_prefix)
        except Exception as e:
            print(f"Error processing {glb_file}: {str(e)}")

json_path = "/home/benzshawelt/.objaverse/hf-objaverse-v1/object-paths.json"
output_dir = "./objaverse_images"
process_objaverse_files(json_path, output_dir)
