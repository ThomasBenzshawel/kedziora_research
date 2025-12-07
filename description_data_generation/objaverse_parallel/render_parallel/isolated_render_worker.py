#!/usr/bin/env python3
"""
Worker script for parallel 3D object rendering on SLURM cluster
Enhanced with process isolation to prevent memory leaks
"""

import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import json
import argparse
import sys
from pathlib import Path
import threading
from datetime import datetime
import shutil
import gc
import multiprocessing as mp
import fcntl
import time

# Global lock for writing to bad items list
bad_items_lock = threading.Lock()


def get_supported_extensions():
    """
    Return a set of supported 3D file extensions.
    """
    return {'.glb', '.gltf', '.obj', '.ply', '.stl', '.dae', '.3ds', '.fbx', '.x3d', '.off'}


def get_bad_items_list_path():
    """
    Get the path for the bad items list file in the directory above the script location.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    return os.path.join(parent_dir, "bad_items_list.json")


def add_to_bad_items_list(uid, file_path, reason, worker_id):
    """
    Safely add an item to the bad items list with file-based locking for multi-process safety.
    """
    bad_items_file = get_bad_items_list_path()
    
    # Create the entry
    entry = {
        "uid": uid,
        "original_file_path": file_path,
        "reason": reason,
        "worker_id": worker_id,
        "timestamp": datetime.now().isoformat(),
        "deleted": False
    }
    
    # Use file-based locking for multi-process safety
    max_retries = 10
    for attempt in range(max_retries):
        try:
            # Open file with exclusive lock
            with open(bad_items_file, 'a+') as f:
                # Acquire exclusive lock (blocks until available)
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                
                try:
                    # Read current contents
                    f.seek(0)
                    content = f.read()
                    
                    if content.strip():
                        bad_items = json.loads(content)
                    else:
                        bad_items = []
                    
                    # Add new entry
                    bad_items.append(entry)
                    
                    # Write back
                    f.seek(0)
                    f.truncate()
                    json.dump(bad_items, f, indent=2, ensure_ascii=False)
                    
                    return True
                finally:
                    # Release lock
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                    
        except (IOError, json.JSONDecodeError) as e:
            if attempt < max_retries - 1:
                time.sleep(0.1 * (attempt + 1))  # Exponential backoff
                continue
            else:
                print(f"Error writing to bad items list after {max_retries} attempts: {e}")
                return False
    
    return False


def update_bad_item_deletion_status(uid, deleted_successfully):
    """
    Update the deletion status with file-based locking.
    """
    bad_items_file = get_bad_items_list_path()
    
    max_retries = 10
    for attempt in range(max_retries):
        try:
            with open(bad_items_file, 'r+') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                
                try:
                    bad_items = json.load(f)
                    
                    # Update the entry
                    for item in bad_items:
                        if item['uid'] == uid:
                            item['deleted'] = deleted_successfully
                            break
                    
                    # Write back
                    f.seek(0)
                    f.truncate()
                    json.dump(bad_items, f, indent=2, ensure_ascii=False)
                    
                    return
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                    
        except (IOError, json.JSONDecodeError) as e:
            if attempt < max_retries - 1:
                time.sleep(0.1 * (attempt + 1))
                continue
            else:
                print(f"Error updating bad items list after {max_retries} attempts: {e}")
                return


def delete_bad_file_and_log(uid, file_path, reason, worker_id):
    """
    Delete a bad 3D model file and add it to the bad items list.
    """
    if not add_to_bad_items_list(uid, file_path, reason, worker_id):
        print(f"Failed to log bad item to list: {uid}")
        return False
    
    deleted_successfully = False
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            deleted_successfully = True
            print(f"Deleted bad file: {file_path} (Reason: {reason})")
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")
            deleted_successfully = False
    else:
        print(f"File not found for deletion: {file_path}")
        deleted_successfully = False
    
    update_bad_item_deletion_status(uid, deleted_successfully)
    return deleted_successfully


def scan_directory_for_3d_files(root_dir, use_folder_name_as_uid=True):
    """
    Scan a directory structure for 3D files and create a mapping with absolute paths.
    """
    root_path = Path(root_dir)
    if not root_path.exists():
        raise ValueError(f"Directory {root_dir} does not exist")
    
    supported_extensions = get_supported_extensions()
    object_paths = {}
    
    print(f"Scanning directory: {root_dir}")
    print(f"Looking for files with extensions: {supported_extensions}")
    
    for file_path in root_path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            if use_folder_name_as_uid:
                uid = file_path.parent.name
                if uid in object_paths:
                    uid = f"{uid}_{file_path.stem}"
            else:
                relative_path = file_path.relative_to(root_path)
                uid = str(relative_path).replace(os.sep, '_').replace('.', '_')
            
            object_paths[uid] = str(file_path.absolute())
    
    print(f"Found {len(object_paths)} 3D files")
    return object_paths


def check_complete_renders(output_dir_path, uid):
    """
    Check if all 6 rendered images exist for a given UID.
    """
    expected_files = ['front', 'back', 'right', 'left', 'up', 'down']
    return all(
        os.path.exists(os.path.join(output_dir_path, f"{uid}_{view}.jpg")) 
        for view in expected_files
    )


def load_object_paths(json_path, scan_directories=None, use_folder_name_as_uid=True):
    """
    Load object paths from JSON file and optionally scan directories for additional files.
    """
    all_object_paths = {}
    
    if json_path and os.path.exists(json_path):
        print(f"Loading object paths from JSON: {json_path}")
        with open(json_path, 'rt', encoding='utf-8') as f:
            json_object_paths = json.load(f)
        
        base_dir = os.path.dirname(json_path)
        for uid, relative_path in json_object_paths.items():
            absolute_path = os.path.abspath(os.path.join(base_dir, relative_path))
            all_object_paths[uid] = absolute_path
            
        print(f"Loaded {len(json_object_paths)} objects from JSON")
    else:
        print(f"JSON file not found or not provided: {json_path}")
    
    if scan_directories:
        for scan_dir in scan_directories:
            if scan_dir and os.path.exists(scan_dir):
                print(f"Scanning directory for 3D files: {scan_dir}")
                try:
                    scanned_object_paths = scan_directory_for_3d_files(scan_dir, use_folder_name_as_uid)
                    
                    for uid, absolute_path in scanned_object_paths.items():
                        if uid in all_object_paths:
                            dir_name = os.path.basename(scan_dir)
                            unique_uid = f"{uid}_{dir_name}"
                            all_object_paths[unique_uid] = absolute_path
                        else:
                            all_object_paths[uid] = absolute_path
                    
                    print(f"Added {len(scanned_object_paths)} objects from directory scan: {scan_dir}")
                except Exception as e:
                    print(f"Error scanning directory {scan_dir}: {e}")
            else:
                print(f"Scan directory not found or not provided: {scan_dir}")
    
    print(f"Total objects to process: {len(all_object_paths)}")
    return all_object_paths


def cleanup_incomplete_renders(output_dir, all_object_paths, worker_id):
    """
    Check previously rendered objects and delete originals with incomplete renders.
    """
    if not os.path.exists(output_dir):
        print("Output directory doesn't exist, skipping cleanup")
        return 0
    
    print("Starting cleanup of incomplete renders...")
    deleted_count = 0
    
    try:
        existing_dirs = [d for d in os.listdir(output_dir) 
                        if os.path.isdir(os.path.join(output_dir, d)) and not d.startswith('worker_')]
    except OSError as e:
        print(f"Error listing output directory: {e}")
        return 0
    
    print(f"Found {len(existing_dirs)} existing output directories to check")
    
    for uid in existing_dirs:
        output_dir_path = os.path.join(output_dir, uid)
        
        if not check_complete_renders(output_dir_path, uid):
            if uid in all_object_paths:
                original_file_path = all_object_paths[uid]
                if os.path.exists(original_file_path):
                    print(f"Cleanup: Found incomplete renders for {uid}, deleting original file")
                    if delete_bad_file_and_log(uid, original_file_path, "incomplete_renders_cleanup", worker_id):
                        deleted_count += 1
                else:
                    print(f"Cleanup: Original file already deleted for {uid}")
            else:
                print(f"Cleanup: Could not find original file path for {uid} (not in current batch)")
            
            try:
                shutil.rmtree(output_dir_path)
                print(f"Cleanup: Removed incomplete output directory for {uid}")
            except Exception as e:
                print(f"Cleanup: Error removing output directory {output_dir_path}: {e}")
    
    print(f"Cleanup completed: {deleted_count} files deleted")
    return deleted_count


def process_single_render(args):
    """
    Process a single 3D file rendering in isolation.
    This function runs in a separate process to prevent memory leaks.
    """
    uid, absolute_file_path, output_dir, check_dir, chunk_id = args
    
    # Import heavy libraries here to avoid sharing memory
    import trimesh
    import pyrender
    import numpy as np
    from PIL import Image
    import cv2
    import gc
    
def is_scene_empty(scene):
    """Check if a loaded scene/mesh is empty or invalid."""
    try:
        if scene is None:
            return True
        
        if hasattr(scene, 'is_empty') and scene.is_empty:
            return True
        
        # Handle single Trimesh objects (no .geometry attribute)
        if isinstance(scene, trimesh.Trimesh):
            if scene.vertices is None or len(scene.vertices) == 0:
                return True
            if scene.faces is None or len(scene.faces) == 0:
                return True
            return False
        
        # Handle Scene objects
        if hasattr(scene, 'geometry'):
            if len(scene.geometry) == 0:
                return True
            
            all_empty = True
            for name, geom in scene.geometry.items():
                if hasattr(geom, 'vertices') and geom.vertices is not None and len(geom.vertices) > 0:
                    all_empty = False
                    break
            if all_empty:
                return True
        
        # Fallback checks for bounds/extents
        if hasattr(scene, 'bounds'):
            bounds = scene.bounds
            if bounds is None or bounds.shape != (2, 3):
                return True
            
            diagonal = bounds[1] - bounds[0]
            if np.allclose(diagonal, 0):
                return True
        
        if hasattr(scene, 'extents'):
            if scene.extents is None or np.allclose(scene.extents, 0):
                return True
        
        return False
        
    except Exception as e:
        print(f"Error checking if scene is empty: {e}")
        return True
    
    def look_at(eye, target):
        """Safe version of look_at function with better error handling"""
        try:
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
                    
                    if not np.allclose(np.linalg.det(rotation_matrix), 1.0, atol=1e-6):
                        continue
                        
                    return rotation_matrix
            
            return np.eye(3)
            
        except Exception as e:
            print(f"Error in look_at function: {e}")
            return np.eye(3)
    
    def get_camera_poses(scene):
        centroid = scene.centroid
        bounds = scene.bounds
        diagonal = np.linalg.norm(bounds[1] - bounds[0])
        
        positions = [
            [0, 0, diagonal],
            [0, 0, -diagonal],
            [diagonal, 0, 0],
            [-diagonal, 0, 0],
            [0, diagonal, 0.075*diagonal],
            [0, -diagonal, 0.075*diagonal],
        ]
        
        camera_poses = []
        for pos in positions:
            pose = np.eye(4)
            pose[:3, 3] = centroid + np.array(pos)
            pose[:3, :3] = look_at(pose[:3, 3], centroid)
            camera_poses.append(pose)
        
        return camera_poses, diagonal
    
    def is_black_and_white(image, color_threshold=10, saturation_threshold=20, value_range_threshold=30):
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hue_std = np.std(hsv[:,:,0])
        sat_mean = np.mean(hsv[:,:,1])
        sat_std = np.std(hsv[:,:,1])
        val_mean = np.mean(hsv[:,:,2])
        val_range = np.max(hsv[:,:,2]) - np.min(hsv[:,:,2])
        
        hist_hue = cv2.calcHist([hsv], [0], None, [36], [0, 180])
        hist_hue = hist_hue / np.sum(hist_hue)
        hue_peaks = np.sum(hist_hue > 0.05)
        
        unique_colors = np.unique(image.reshape(-1, 3), axis=0)
        
        is_bw = (
            (hue_std < 5) and
            (sat_mean < saturation_threshold) and
            (sat_std < saturation_threshold/2) and
            (hue_peaks <= 2) and
            (len(unique_colors) <= color_threshold or 
             (val_range < value_range_threshold and sat_mean < 10))
        )
        
        if is_bw:
            if val_mean < 50:
                return "black"
            else:
                return "white"
        else:
            return "color"
    
    # Main rendering logic
    scene = None
    pyrender_scene = None
    r = None
    
    try:
        # Check if file exists
        if not os.path.exists(absolute_file_path):
            return 'error', uid, "File not found"
        
        # Check if already complete
        output_dir_path = os.path.join(output_dir, uid)
        if os.path.exists(output_dir_path) and check_complete_renders(output_dir_path, uid):
            return 'skipped', uid, None
        
        # Create output directory
        os.makedirs(output_dir_path, exist_ok=True)
        output_prefix = os.path.join(output_dir_path, uid)
        
        # Load scene
        scene = trimesh.load(absolute_file_path)

        # Convert single Trimesh to Scene for consistent handling
        if isinstance(scene, trimesh.Trimesh):
            mesh = scene
            scene = trimesh.Scene()
            scene.add_geometry(mesh)

        # Check if empty
        if is_scene_empty(scene):
            return 'empty', uid, None
        
        # Create pyrender scene
        pyrender_scene = pyrender.Scene.from_trimesh_scene(
            scene, bg_color=[0.6, 0.6, 0.6], ambient_light=[1, 1, 1]
        )
        
        camera_poses, scene_diagonal = get_camera_poses(scene)
        camera_distance = scene_diagonal * 1.15
        camera_distance = max(camera_distance, 0.1)
        camera_distance = min(camera_distance, 300.0)
        fov = min(2 * np.arctan(scene_diagonal / (2 * camera_distance)), np.pi/2)
        
        perspectives = ['front', 'back', 'right', 'left', 'up', 'down']
        
        pyrender_scene.add(pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=0.5), pose=np.eye(4))
        camera = pyrender.PerspectiveCamera(yfov=fov)
        r = pyrender.OffscreenRenderer(viewport_width=800, viewport_height=600, point_size=1.0)
        
        for pos in camera_poses:
            pyrender_scene.add(pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=0.5), pose=pos)
            light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2)
            pyrender_scene.add(light, pose=pos)
        
        rendered_count = 0
        for i, pose in enumerate(camera_poses):
            light_intensity = 2
            light = pyrender.SpotLight(
                color=np.ones(3), intensity=light_intensity,
                innerConeAngle=np.pi/8.0, outerConeAngle=np.pi/4.0
            )
            light_node = pyrender_scene.add(light, pose=pose)
            
            image_is_problematic = False
            color = None
            
            for iteration in range(10):
                camera_node = pyrender_scene.add(camera, pose=pose)
                color, _ = r.render(pyrender_scene)
                is_bw = is_black_and_white(color)
                pyrender_scene.remove_node(camera_node)
                
                if is_bw == "color":
                    break
                
                if iteration < 9:
                    if is_bw == "black":
                        light_intensity *= 1.75
                    elif is_bw == "white":
                        light_intensity *= 0.5
                    pyrender_scene.remove_node(light_node)
                    light = pyrender.SpotLight(
                        color=np.ones(3), intensity=light_intensity,
                        innerConeAngle=np.pi/8.0, outerConeAngle=np.pi/4.0
                    )
                    light_node = pyrender_scene.add(light, pose=pose)
                else:
                    image_is_problematic = True
                    break
            
            if color is not None:
                img = Image.fromarray(color)
                
                if image_is_problematic and check_dir is not None:
                    filename = f"{os.path.basename(absolute_file_path).split('.')[0]}_{perspectives[i]}.jpg"
                    output_path = os.path.join(check_dir, filename)
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    img.save(output_path, 'JPEG', quality=95)
                else:
                    output_path = f"{output_prefix}_{perspectives[i]}.jpg"
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    img.save(output_path, 'JPEG', quality=95)
                    rendered_count += 1
                
                del img
            
            pyrender_scene.remove_node(light_node)
            
            if i % 3 == 0:
                gc.collect()
        
        # Check if all images were created
        if rendered_count == 6 and check_complete_renders(output_dir_path, uid):
            return 'success', uid, None
        else:
            return 'incomplete', uid, None
            
    except Exception as e:
        return 'error', uid, str(e)
    finally:
        if r is not None:
            try:
                r.delete()
            except:
                pass
        
        if pyrender_scene is not None:
            del pyrender_scene
        if scene is not None:
            del scene
        
        gc.collect()


def get_work_chunk(object_paths, chunk_id, total_chunks):
    """Split the work into chunks for parallel processing"""
    items = list(object_paths.items())
    chunk_size = len(items) // total_chunks
    start_idx = chunk_id * chunk_size
    
    if chunk_id == total_chunks - 1:
        end_idx = len(items)
    else:
        end_idx = start_idx + chunk_size
    
    return dict(items[start_idx:end_idx])


def process_chunk(json_path, output_dir, chunk_id, total_chunks, scan_dir=None, scan_dir_2=None,
                 use_folder_name_as_uid=True, log_file=None, check_dir=None):
    """Process a chunk of the 3D dataset with process isolation"""
    
    # Set up logging
    if log_file:
        sys.stdout = open(log_file, 'w')
        sys.stderr = sys.stdout
    
    print(f"Worker {chunk_id}/{total_chunks} starting...")
    print(f"GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Unknown')}")
    print(f"Process isolation: Enabled (batch size: 25)")
    
    # Prepare directories to scan
    scan_directories = []
    if scan_dir:
        scan_directories.append(scan_dir)
    if scan_dir_2:
        if not scan_dir or os.path.abspath(scan_dir_2) != os.path.abspath(scan_dir):
            scan_directories.append(scan_dir_2)
    
    # Load object paths
    all_object_paths = load_object_paths(json_path, scan_directories, use_folder_name_as_uid)
    
    if not all_object_paths:
        print("No objects found to process!")
        return
    
    # Only worker 0 performs cleanup
    cleanup_deleted = 0
    if chunk_id == 0:
        cleanup_deleted = cleanup_incomplete_renders(output_dir, all_object_paths, chunk_id)
    
    # Get this worker's chunk
    object_paths = get_work_chunk(all_object_paths, chunk_id, total_chunks)
    print(f"Worker {chunk_id} processing {len(object_paths)} objects")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare arguments for multiprocessing
    args_list = [
        (uid, absolute_file_path, output_dir, check_dir, chunk_id)
        for uid, absolute_file_path in object_paths.items()
    ]
    
    # Create lookup dictionary for efficient path retrieval
    uid_to_path = {item[0]: item[1] for item in args_list}
    
    del object_paths
    del all_object_paths
    gc.collect()
    
    # Statistics
    processed = 0
    errors = 0
    empty_scenes = 0
    deleted_files = 0
    skipped = 0
    
    # Process with isolated workers (smaller batches due to rendering overhead)
    batch_size = 25  # Smaller batch for rendering
    total_batches = (len(args_list) + batch_size - 1) // batch_size
    
    print(f"Processing {len(args_list)} files in {total_batches} batches of up to {batch_size} files each")
    
    import time
    start_time = time.time()
    
    for batch_idx in range(total_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(args_list))
        batch = args_list[batch_start:batch_end]
        
        print(f"Starting batch {batch_idx + 1}/{total_batches} ({len(batch)} files)...")
        
        # Create fresh pool for each batch
        with mp.Pool(processes=1) as pool:
            results = pool.map(process_single_render, batch)
        
        # Process results
        for status, uid, info in results:
            if status == 'success':
                processed += 1
            elif status == 'skipped':
                skipped += 1
            elif status == 'empty':
                empty_scenes += 1
                # Delete empty scene file
                original_path = uid_to_path[uid]
                if delete_bad_file_and_log(uid, original_path, "empty_scene", chunk_id):
                    deleted_files += 1
            elif status == 'incomplete':
                errors += 1
                original_path = uid_to_path[uid]
                if delete_bad_file_and_log(uid, original_path, "incomplete_renders", chunk_id):
                    deleted_files += 1
            elif status == 'error':
                errors += 1
                print(f"Error processing {uid}: {info}")
                original_path = uid_to_path[uid]
                if delete_bad_file_and_log(uid, original_path, "render_error", chunk_id):
                    deleted_files += 1
        
        # Progress logging
        elapsed = time.time() - start_time
        items_done = batch_end
        rate = items_done / elapsed if elapsed > 0 else 0
        eta = (len(args_list) - items_done) / rate if rate > 0 else 0
        
        print(f"Batch {batch_idx + 1}/{total_batches} complete. "
              f"Progress: {items_done}/{len(args_list)} "
              f"(Processed: {processed}, Skipped: {skipped}, Empty: {empty_scenes}, Errors: {errors}) "
              f"Rate: {rate:.2f} items/s, ETA: {eta/60:.1f} min")
        
        gc.collect()
    
    elapsed_time = time.time() - start_time
    print(f"Worker {chunk_id} completed in {elapsed_time/60:.2f} minutes")
    print(f"Processed: {processed}, Skipped: {skipped}, Errors: {errors}, Empty scenes: {empty_scenes}, Deleted: {deleted_files}")
    if chunk_id == 0 and cleanup_deleted > 0:
        print(f"Cleanup: {cleanup_deleted} incomplete renders cleaned up")
    
    # Write completion status
    status_file = os.path.join(output_dir, f"worker_{chunk_id}_status.txt")
    with open(status_file, 'w') as f:
        f.write(f"processed: {processed}\n")
        f.write(f"errors: {errors}\n")
        f.write(f"empty_scenes: {empty_scenes}\n")
        f.write(f"deleted_files: {deleted_files}\n")
        f.write(f"cleanup_deleted: {cleanup_deleted}\n")
        f.write(f"skipped: {skipped}\n")
        f.write(f"total_assigned: {len(args_list)}\n")


def main():
    parser = argparse.ArgumentParser(description='Render 3D objects in parallel with process isolation')
    parser.add_argument('--json_path', help='Path to object-paths.json (optional)')
    parser.add_argument('--scan_dir', help='Directory to scan for 3D files (optional)')
    parser.add_argument('--scan_dir_2', help='Second directory to scan for 3D files (optional)')
    parser.add_argument('--output_dir', required=True, help='Output directory for images')
    parser.add_argument('--chunk_id', type=int, required=True, help='Chunk ID (0-indexed)')
    parser.add_argument('--total_chunks', type=int, required=True, help='Total number of chunks')
    parser.add_argument('--use_folder_name_as_uid', action='store_true', default=True,
                       help='Use folder name as UID when scanning directory')
    parser.add_argument('--log_file', help='Log file path')
    parser.add_argument('--check_dir', help='Directory to put problematic files', 
                       default="../check_objaverse_images/")

    args = parser.parse_args()
    
    if not args.json_path and not args.scan_dir and not args.scan_dir_2:
        print("Error: Must provide either --json_path or --scan_dir (or both)")
        sys.exit(1)
    
    process_chunk(args.json_path, args.output_dir, args.chunk_id, 
                 args.total_chunks, args.scan_dir, args.scan_dir_2, 
                 args.use_folder_name_as_uid, args.log_file, args.check_dir)


if __name__ == "__main__":
    main()