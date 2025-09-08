#!/usr/bin/env python3
"""
Worker script for parallel 3D object rendering on SLURM cluster
Enhanced to support both JSON file and directory scanning for 3D files
Added empty scene detection to skip invalid/empty 3D models
"""

import trimesh
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import pyrender
import numpy as np
from PIL import Image
import cv2
import json
import os
from tqdm import tqdm
import time
import argparse
import sys
from pathlib import Path

def get_supported_extensions():
    """
    Return a set of supported 3D file extensions.
    """
    return {'.glb', '.gltf', '.obj', '.ply', '.stl', '.dae', '.3ds', '.fbx', '.x3d', '.off'}

def scan_directory_for_3d_files(root_dir, use_folder_name_as_uid=True):
    """
    Scan a directory structure for 3D files and create a mapping similar to the JSON format.
   
    Args:
        root_dir (str): Root directory to scan
        use_folder_name_as_uid (bool): If True, use the immediate parent folder name as UID.
                                     If False, generate UID from file path.
   
    Returns:
        dict: Mapping of UIDs to file paths
    """
    root_path = Path(root_dir)
    if not root_path.exists():
        raise ValueError(f"Directory {root_dir} does not exist")
   
    supported_extensions = get_supported_extensions()
    object_paths = {}
   
    print(f"Scanning directory: {root_dir}")
    print(f"Looking for files with extensions: {supported_extensions}")
   
    # Find all 3D files recursively
    for file_path in root_path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            if use_folder_name_as_uid:
                # Use the immediate parent folder name as UID
                uid = file_path.parent.name
               
                # If multiple files in same folder, append file stem to make unique
                if uid in object_paths:
                    uid = f"{uid}_{file_path.stem}"
            else:
                # Generate UID from relative path (replace separators with underscores)
                relative_path = file_path.relative_to(root_path)
                uid = str(relative_path).replace(os.sep, '_').replace('.', '_')
           
            # Store relative path from root directory
            relative_file_path = file_path.relative_to(root_path)
            object_paths[uid] = str(relative_file_path)
   
    print(f"Found {len(object_paths)} 3D files")
    return object_paths

def is_scene_empty(scene):
    """
    Check if a loaded scene is empty or invalid.
    
    Args:
        scene: Trimesh scene object
        
    Returns:
        bool: True if scene is empty/invalid, False otherwise
    """
    try:
        # Check if scene is None
        if scene is None:
            return True
        
        # Check if it's a Scene object with no geometry
        if hasattr(scene, 'is_empty') and scene.is_empty:
            return True
        
        # Check if it has any vertices
        if hasattr(scene, 'vertices'):
            if scene.vertices is None or len(scene.vertices) == 0:
                return True
        
        # For Scene objects, check if there are any geometries
        if hasattr(scene, 'geometry'):
            if len(scene.geometry) == 0:
                return True
            
            # Check if all geometries are empty
            all_empty = True
            for name, geom in scene.geometry.items():
                if hasattr(geom, 'vertices') and geom.vertices is not None and len(geom.vertices) > 0:
                    all_empty = False
                    break
            if all_empty:
                return True
        
        # Check bounds to see if they're valid
        if hasattr(scene, 'bounds'):
            bounds = scene.bounds
            if bounds is None or bounds.shape != (2, 3):
                return True
            
            # Check if bounds are degenerate (zero volume)
            diagonal = bounds[1] - bounds[0]
            if np.allclose(diagonal, 0):
                return True
        
        # Check if the scene has zero extents
        if hasattr(scene, 'extents'):
            if scene.extents is None or np.allclose(scene.extents, 0):
                return True
        
        return False
        
    except Exception as e:
        print(f"Error checking if scene is empty: {e}")
        return True  # Consider it empty if we can't check properly

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
    """Safe version of look_at function with better error handling"""
    try:
        # Validate inputs
        if eye is None or target is None:
            return np.eye(3)
        
        eye = np.array(eye)
        target = np.array(target)
        
        if eye.shape != (3,) or target.shape != (3,):
            return np.eye(3)
        
        forward = target - eye
        forward_norm = np.linalg.norm(forward)
        
        if forward_norm < 1e-10:
            return np.eye(3)
            
        forward = forward / forward_norm
        
        # Try different up vectors if one fails
        up_vectors = [
            np.array([0, 1, 0]),
            np.array([1, 0, 0]),
            np.array([0, 0, 1])
        ]
        
        for up_vector in up_vectors:
            to_side = np.cross(forward, up_vector)
            to_side_norm = np.linalg.norm(to_side)
            
            if to_side_norm > 1e-10:
                to_side = to_side / to_side_norm
                up = np.cross(to_side, forward)
                rotation_matrix = np.column_stack((to_side, up, -forward))
                
                # Validate the rotation matrix
                if not np.allclose(np.linalg.det(rotation_matrix), 1.0, atol=1e-6):
                    continue
                    
                return rotation_matrix
        
        # If all else fails, return identity
        return np.eye(3)
        
    except Exception as e:
        print(f"Error in look_at function: {e}")
        return np.eye(3)

def render_glb(glb_path, output_prefix, resolution=(800, 600), check_dir=None):
    """
    Render a 3D file from multiple viewpoints.
    
    Args:
        glb_path: Path to the 3D file
        output_prefix: Prefix for output image files
        resolution: Output image resolution
        check_dir: Directory to save problematic images
        
    Returns:
        bool: True if successfully rendered, False if scene was empty or error occurred
    """
    try:
        scene = trimesh.load(glb_path)
        
        # Check if the scene is empty
        if is_scene_empty(scene):
            print(f"WARNING: Scene is empty or invalid for file: {glb_path}")
            print(f"Skipping rendering for empty scene: {os.path.basename(glb_path)}")
            return False

        pyrender_scene = pyrender.Scene.from_trimesh_scene(scene, bg_color=[0.6, 0.6, 0.6], ambient_light = [1, 1, 1])
        
        camera_poses, scene_diagonal = get_camera_poses(scene)
        camera_distance = scene_diagonal * 1.15 # add a 15% margin
        camera_distance = max(camera_distance, 0.1)  # Prevent division by zero
        camera_distance = min(camera_distance, 300.0)  # Cap maximum distance
        fov = min(2 * np.arctan(scene_diagonal / (2 * camera_distance)), np.pi/2)  # Cap FOV

        perspectives = ['front', 'back', 'right', 'left', 'up', 'down']
        
        # add a "sun" point light to the scene
        pyrender_scene.add(pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=0.5), pose=np.eye(4))

        camera = pyrender.PerspectiveCamera(yfov=fov)

        r = pyrender.OffscreenRenderer(viewport_width=resolution[0],
            viewport_height=resolution[1], point_size=1.0)

        for pos in camera_poses:
            pyrender_scene.add(pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=0.5), pose=pos)
            light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2)
            pyrender_scene.add(light, pose=pos)

        for i, pose in enumerate(camera_poses):
            light_intensity = 2  # Initial light intensity
            light = pyrender.SpotLight(color=np.ones(3), intensity=light_intensity,
                                                innerConeAngle=np.pi/8.0,
                                                outerConeAngle=np.pi/4.0)
            light_node = pyrender_scene.add(light, pose=pose)

            image_is_problematic = False
            color = None
            
            for iteration in range(10):
                camera_node = pyrender_scene.add(camera, pose=pose)

                color, _ = r.render(pyrender_scene)
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
                    # If we've reached 10 iterations, mark as problematic
                    print(f"Warning: {perspectives[i]} view remained black and white after 10 iterations.")
                    image_is_problematic = True
                    break

            # Save the image
            if color is not None:
                img = Image.fromarray(color)
                
                if image_is_problematic and check_dir is not None:
                    # Save problematic image to check directory
                    filename = f"{os.path.basename(glb_path).split('.')[0]}_{perspectives[i]}.jpg"
                    output_path = os.path.join(check_dir, filename)
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    img.save(output_path, 'JPEG', quality=95)
                    print(f"Saved problematic image to: {output_path}")
                else:
                    # Save normal image to output directory
                    output_path = f"{output_prefix}_{perspectives[i]}.jpg"
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    img.save(output_path, 'JPEG', quality=95)

            pyrender_scene.remove_node(light_node)
        
        return True
        
    except Exception as e:
        print(f"Error rendering {glb_path}: {str(e)}")
        return False
    

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

def get_work_chunk(object_paths, chunk_id, total_chunks):
    """Split the work into chunks for parallel processing"""
    items = list(object_paths.items())
    chunk_size = len(items) // total_chunks
    start_idx = chunk_id * chunk_size
    
    if chunk_id == total_chunks - 1:
        # Last chunk gets any remaining items
        end_idx = len(items)
    else:
        end_idx = start_idx + chunk_size
    
    return dict(items[start_idx:end_idx])

def load_object_paths(json_path, scan_dir=None, use_folder_name_as_uid=True):
    """
    Load object paths from JSON file and optionally scan directory for additional files.
    
    Args:
        json_path (str): Path to JSON file with object paths
        scan_dir (str, optional): Directory to scan for additional 3D files
        use_folder_name_as_uid (bool): Whether to use folder names as UIDs when scanning
        
    Returns:
        dict: Combined mapping of UIDs to file paths
        str: Base directory for resolving relative paths
    """
    all_object_paths = {}
    base_dir = None
    
    # First, load from JSON if it exists
    if json_path and os.path.exists(json_path):
        print(f"Loading object paths from JSON: {json_path}")
        with open(json_path, 'rt', encoding='utf-8') as f:
            json_object_paths = json.load(f)
        all_object_paths.update(json_object_paths)
        base_dir = os.path.dirname(json_path)
        print(f"Loaded {len(json_object_paths)} objects from JSON")
    else:
        print(f"JSON file not found or not provided: {json_path}")
    
    # Then, scan directory for additional files if provided
    if scan_dir and os.path.exists(scan_dir):
        print(f"Scanning directory for additional 3D files: {scan_dir}")
        try:
            scanned_object_paths = scan_directory_for_3d_files(scan_dir, use_folder_name_as_uid)
            all_object_paths.update(scanned_object_paths)
            
            # If no base_dir from JSON, use scan_dir as base
            if base_dir is None:
                base_dir = scan_dir
                
            print(f"Added {len(scanned_object_paths)} objects from directory scan")
        except Exception as e:
            print(f"Error scanning directory {scan_dir}: {e}")
    else:
        print(f"Scan directory not found or not provided: {scan_dir}")
    
    print(f"Total objects to process: {len(all_object_paths)}")
    return all_object_paths, base_dir

def process_chunk(json_path, output_dir, chunk_id, total_chunks, scan_dir=None, 
                 use_folder_name_as_uid=True, log_file=None, check_dir=None):
    """Process a chunk of the 3D dataset from JSON and/or directory scanning"""
    
    # Set up logging
    if log_file:
        sys.stdout = open(log_file, 'w')
        sys.stderr = sys.stdout
    
    print(f"Worker {chunk_id}/{total_chunks} starting...")
    print(f"GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Unknown')}")
    
    # Load object paths from JSON and/or directory scanning
    all_object_paths, base_dir = load_object_paths(json_path, scan_dir, use_folder_name_as_uid)
    
    if not all_object_paths:
        print("No objects found to process!")
        return
    
    # Get this worker's chunk
    object_paths = get_work_chunk(all_object_paths, chunk_id, total_chunks)
    
    print(f"Worker {chunk_id} processing {len(object_paths)} objects")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    processed = 0
    errors = 0
    empty_scenes = 0
    
    # Process objects in this chunk
    for uid, file_path in tqdm(object_paths.items(), 
                              desc=f"Worker {chunk_id}"):
        # Construct the full path to the 3D file
        if base_dir:
            full_file_path = os.path.join(base_dir, file_path)
        else:
            full_file_path = file_path
        
        # Check if the file exists
        if not os.path.exists(full_file_path):
            print(f"File not found: {full_file_path}")
            errors += 1
            continue

        # Skip if already processed
        if os.path.exists(os.path.join(output_dir, uid)):
            # Check if the directory has actual rendered images
            output_dir_path = os.path.join(output_dir, uid)
            expected_files = ['front', 'back', 'right', 'left', 'up', 'down']
            all_files_exist = all(
                os.path.exists(os.path.join(output_dir_path, f"{uid}_{view}.jpg")) 
                for view in expected_files
            )
            
            if all_files_exist:
                print(f"Directory already exists with complete renders for {uid}, skipping.")
                continue
            else:
                print(f"Directory exists but incomplete renders for {uid}, re-processing.")
        else:
            os.makedirs(os.path.join(output_dir, uid), exist_ok=True)
        
        # Create an output prefix for this object 
        output_prefix = os.path.join(output_dir, uid, uid)
        
        try:
            # Render the 3D file
            success = render_glb(full_file_path, output_prefix, check_dir=check_dir)
            if success:
                processed += 1
            else:
                empty_scenes += 1
        except Exception as e:
            print(f"Error processing {full_file_path}: {str(e)}")
            errors += 1
    
    print(f"Worker {chunk_id} completed: {processed} processed, {errors} errors, {empty_scenes} empty scenes skipped")
    
    # Write completion status
    status_file = os.path.join(output_dir, f"worker_{chunk_id}_status.txt")
    with open(status_file, 'w') as f:
        f.write(f"processed: {processed}\n")
        f.write(f"errors: {errors}\n")
        f.write(f"empty_scenes: {empty_scenes}\n")
        f.write(f"total_assigned: {len(object_paths)}\n")

def main():
    parser = argparse.ArgumentParser(description='Render 3D objects in parallel from JSON and/or directory scanning')
    parser.add_argument('--json_path', help='Path to object-paths.json (optional)')
    parser.add_argument('--scan_dir', help='Directory to scan for 3D files (optional)')
    parser.add_argument('--output_dir', required=True, help='Output directory for images')
    parser.add_argument('--chunk_id', type=int, required=True, help='Chunk ID (0-indexed)')
    parser.add_argument('--total_chunks', type=int, required=True, help='Total number of chunks')
    parser.add_argument('--use_folder_name_as_uid', action='store_true', default=True,
                       help='Use folder name as UID when scanning directory')
    parser.add_argument('--log_file', help='Log file path')
    parser.add_argument('--check_dir', help='Directory to put problematic files', default="../check_objaverse_images/")

    args = parser.parse_args()
    
    # Validate that at least one source is provided
    if not args.json_path and not args.scan_dir:
        print("Error: Must provide either --json_path or --scan_dir (or both)")
        sys.exit(1)
    
    process_chunk(args.json_path, args.output_dir, args.chunk_id, 
                 args.total_chunks, args.scan_dir, args.use_folder_name_as_uid, 
                 args.log_file, args.check_dir)

if __name__ == "__main__":
    main()