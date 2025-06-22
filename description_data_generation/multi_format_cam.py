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
import subprocess
import tempfile
from pathlib import Path

# Optional imports for extended format support
try:
    import bpy  # Blender Python API
    HAS_BLENDER = True
except ImportError:
    HAS_BLENDER = False

try:
    from pxr import Usd, UsdGeom  # USD support
    HAS_USD = True
except ImportError:
    HAS_USD = False

VERIFICATION = "./check_objaverse_images/"

class MultiFormatLoader:
    """Handles loading of various 3D file formats"""
    
    @staticmethod
    def load_scene(file_path):
        """Load a 3D scene from various file formats"""
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        
        # Try trimesh first for supported formats
        if extension in ['.obj', '.glb', '.gltf', '.stl', '.dae', '.ply']:
            return MultiFormatLoader._load_with_trimesh(file_path)
        
        # Handle special formats
        elif extension == '.fbx':
            return MultiFormatLoader._load_fbx(file_path)
        
        elif extension in ['.usd', '.usda', '.usdz']:
            return MultiFormatLoader._load_usd(file_path)
        
        elif extension == '.abc':
            return MultiFormatLoader._load_alembic(file_path)
        
        elif extension == '.blend':
            return MultiFormatLoader._load_blend(file_path)
        
        else:
            # Fallback: try trimesh anyway (it might support more than documented)
            return MultiFormatLoader._load_with_trimesh(file_path)
    
    @staticmethod
    def _load_with_trimesh(file_path):
        """Load using trimesh (handles most formats)"""
        try:
            scene = trimesh.load(str(file_path))
            if hasattr(scene, 'geometry') and len(scene.geometry) == 0:
                raise ValueError("Empty scene")
            return scene
        except Exception as e:
            print(f"Trimesh failed to load {file_path}: {e}")
            return None
    
    @staticmethod
    def _load_fbx(file_path):
        """Load FBX files (requires assimp or conversion)"""
        try:
            # Try with trimesh first (if assimp is installed)
            scene = trimesh.load(str(file_path))
            return scene
        except:
            # Fallback: convert to OBJ using Blender if available
            if HAS_BLENDER:
                return MultiFormatLoader._convert_with_blender(file_path, '.obj')
            else:
                print(f"Cannot load FBX {file_path}: install assimp or Blender")
                return None
    
    @staticmethod
    def _load_usd(file_path):
        """Load USD/USDA/USDZ files"""
        if not HAS_USD:
            print(f"Cannot load USD {file_path}: install USD Python bindings")
            return None
        
        try:
            # Convert USD to OBJ using USD tools
            temp_obj = MultiFormatLoader._usd_to_obj(file_path)
            if temp_obj:
                scene = trimesh.load(temp_obj)
                os.unlink(temp_obj)  # Clean up temp file
                return scene
        except Exception as e:
            print(f"Failed to load USD {file_path}: {e}")
        
        return None
    
    @staticmethod
    def _load_alembic(file_path):
        """Load Alembic (.abc) files"""
        # Alembic requires specialized tools, try Blender conversion
        if HAS_BLENDER:
            return MultiFormatLoader._convert_with_blender(file_path, '.obj')
        else:
            print(f"Cannot load Alembic {file_path}: install Blender")
            return None
    
    @staticmethod
    def _load_blend(file_path):
        """Load Blender (.blend) files"""
        if HAS_BLENDER:
            return MultiFormatLoader._convert_with_blender(file_path, '.obj')
        else:
            # Try command-line Blender if available
            return MultiFormatLoader._convert_with_blender_cli(file_path)
    
    @staticmethod
    def _convert_with_blender(file_path, target_format='.obj'):
        """Convert files using Blender Python API"""
        try:
            # Clear existing scene
            bpy.ops.object.select_all(action='SELECT')
            bpy.ops.object.delete(use_global=False)
            
            # Import the file
            if file_path.suffix.lower() == '.fbx':
                bpy.ops.import_scene.fbx(filepath=str(file_path))
            elif file_path.suffix.lower() == '.blend':
                bpy.ops.wm.open_mainfile(filepath=str(file_path))
            elif file_path.suffix.lower() == '.abc':
                bpy.ops.wm.alembic_import(filepath=str(file_path))
            
            # Export to OBJ
            temp_obj = tempfile.mktemp(suffix='.obj')
            bpy.ops.export_scene.obj(filepath=temp_obj)
            
            # Load with trimesh
            scene = trimesh.load(temp_obj)
            os.unlink(temp_obj)
            return scene
            
        except Exception as e:
            print(f"Blender conversion failed for {file_path}: {e}")
            return None
    
    @staticmethod
    def _convert_with_blender_cli(file_path):
        """Convert files using command-line Blender"""
        try:
            temp_obj = tempfile.mktemp(suffix='.obj')
            
            # Blender script to convert file
            script = f"""
import bpy
import sys

# Clear scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# Import file
try:
    if '{file_path.suffix.lower()}' == '.blend':
        bpy.ops.wm.open_mainfile(filepath='{file_path}')
    elif '{file_path.suffix.lower()}' == '.fbx':
        bpy.ops.import_scene.fbx(filepath='{file_path}')
    elif '{file_path.suffix.lower()}' == '.abc':
        bpy.ops.wm.alembic_import(filepath='{file_path}')
    
    # Export to OBJ
    bpy.ops.export_scene.obj(filepath='{temp_obj}')
    print("SUCCESS")
except Exception as e:
    print(f"ERROR: {{e}}")
    sys.exit(1)
"""
            
            script_file = tempfile.mktemp(suffix='.py')
            with open(script_file, 'w') as f:
                f.write(script)
            
            # Run Blender
            result = subprocess.run([
                'blender', '--background', '--python', script_file
            ], capture_output=True, text=True, timeout=30)
            
            os.unlink(script_file)
            
            if result.returncode == 0 and os.path.exists(temp_obj):
                scene = trimesh.load(temp_obj)
                os.unlink(temp_obj)
                return scene
            else:
                print(f"Blender CLI conversion failed: {result.stderr}")
                return None
                
        except Exception as e:
            print(f"Blender CLI failed for {file_path}: {e}")
            return None
    
    @staticmethod
    def _usd_to_obj(file_path):
        """Convert USD to OBJ using USD tools"""
        try:
            from pxr import Usd, UsdGeom, Gf
            
            # Open USD stage
            stage = Usd.Stage.Open(str(file_path))
            if not stage:
                return None
            
            temp_obj = tempfile.mktemp(suffix='.obj')
            
            # Simple USD to OBJ conversion (basic implementation)
            # For production, consider using USD's own export tools
            with open(temp_obj, 'w') as f:
                f.write("# Converted from USD\n")
                
                vertex_count = 0
                for prim in stage.TraverseAll():
                    if prim.IsA(UsdGeom.Mesh):
                        mesh = UsdGeom.Mesh(prim)
                        points = mesh.GetPointsAttr().Get()
                        if points:
                            for point in points:
                                f.write(f"v {point[0]} {point[1]} {point[2]}\n")
                            
                            # Write faces (simplified)
                            face_vertex_counts = mesh.GetFaceVertexCountsAttr().Get()
                            face_vertex_indices = mesh.GetFaceVertexIndicesAttr().Get()
                            
                            if face_vertex_counts and face_vertex_indices:
                                idx = 0
                                for count in face_vertex_counts:
                                    if count == 3:  # Triangle
                                        f.write(f"f {face_vertex_indices[idx]+1+vertex_count} "
                                               f"{face_vertex_indices[idx+1]+1+vertex_count} "
                                               f"{face_vertex_indices[idx+2]+1+vertex_count}\n")
                                    idx += count
                            
                            vertex_count += len(points)
            
            return temp_obj if os.path.getsize(temp_obj) > 50 else None  # Basic sanity check
            
        except Exception as e:
            print(f"USD conversion failed: {e}")
            return None

def render_glb(file_path, output_prefix, resolution=(800, 600)):
    """Updated render function that handles multiple formats"""
    # Load scene using the multi-format loader
    scene = MultiFormatLoader.load_scene(file_path)
    
    if scene is None:
        raise ValueError(f"Could not load scene from {file_path}")
    
    # Handle coordinate system differences
    scene = normalize_scene_orientation(scene)
    
    # Rest of your original render_glb function...
    pyrender_scene = pyrender.Scene.from_trimesh_scene(scene, bg_color=[0.6, 0.6, 0.6], ambient_light=[1, 1, 1])
    
    camera_poses, scene_diagonal = get_camera_poses(scene)
    # Fixed FOV calculation
    camera_distance = scene_diagonal
    fov = 2 * np.arctan(scene_diagonal / (2 * camera_distance))
    camera = pyrender.PerspectiveCamera(yfov=fov)

    perspectives = ['front', 'back', 'right', 'left', 'up', 'down']
    
    # Create renderer once
    r = pyrender.OffscreenRenderer(viewport_width=resolution[0], viewport_height=resolution[1])
    
    try:
        for i, pose in enumerate(camera_poses):
            light_intensity = 2
            light = pyrender.SpotLight(color=np.ones(3), intensity=light_intensity,
                                     innerConeAngle=np.pi/8.0, outerConeAngle=np.pi/4.0)
            light_node = pyrender_scene.add(light, pose=pose)

            for iteration in range(10):
                camera_node = pyrender_scene.add(camera, pose=pose)
                color, _ = r.render(pyrender_scene)
                is_bw = is_black_and_white(color)
                pyrender_scene.remove_node(camera_node)
                
                if is_bw == "color":
                    break
                
                if iteration < 9:
                    if is_bw == "black":
                        print(f"Image is black, increasing light intensity for {perspectives[i]} view.")
                        light_intensity *= 1.75
                        pyrender_scene.remove_node(light_node)
                        light = pyrender.SpotLight(color=np.ones(3), intensity=light_intensity,
                                                innerConeAngle=np.pi/8.0, outerConeAngle=np.pi/4.0)
                        light_node = pyrender_scene.add(light, pose=pose)
                    elif is_bw == "white":
                        print(f"Image is white, decreasing light intensity for {perspectives[i]} view.")
                        light_intensity *= 0.5
                        pyrender_scene.remove_node(light_node)
                        light = pyrender.SpotLight(color=np.ones(3), intensity=light_intensity,
                                                innerConeAngle=np.pi/8.0, outerConeAngle=np.pi/4.0)
                        light_node = pyrender_scene.add(light, pose=pose)
                else:
                    print(f"Warning: {perspectives[i]} view remained black and white after 10 iterations.")

            img = Image.fromarray(color)
            output_path = f"{output_prefix}_{perspectives[i]}.jpg"
            img.save(output_path, 'JPEG', quality=95)
            pyrender_scene.remove_node(light_node)
    
    finally:
        r.delete()

def normalize_scene_orientation(scene):
    """Normalize scene orientation for consistent rendering across formats"""
    # Some formats have different coordinate systems
    # This function helps standardize them
    
    if hasattr(scene, 'bounds'):
        # Ensure scene is reasonably sized
        bounds = scene.bounds
        size = np.linalg.norm(bounds[1] - bounds[0])
        
        # Scale if too small or too large
        if size < 0.1 or size > 100:
            scale_factor = 1.0 / size
            scene.apply_scale(scale_factor)
    
    return scene

def get_supported_extensions():
    """Return list of supported file extensions"""
    base_extensions = ['.obj', '.glb', '.gltf', '.stl', '.dae', '.ply']
    
    if HAS_BLENDER:
        base_extensions.extend(['.fbx', '.abc', '.blend'])
    
    if HAS_USD:
        base_extensions.extend(['.usd', '.usda', '.usdz'])
    
    return base_extensions

def process_objaverse_files(json_path, output_dir):
    """Updated to handle multiple file formats"""
    with open(json_path, 'rt', encoding='utf-8') as f:
        object_paths = json.load(f)
    
    os.makedirs(output_dir, exist_ok=True)
    supported_extensions = get_supported_extensions()
    
    processed = 0
    skipped = 0
    
    for uid, file_path in tqdm(object_paths.items(), desc="Processing files"):
        file_full_path = os.path.join(os.path.dirname(json_path), file_path)
        
        # Check if file exists
        if not os.path.exists(file_full_path):
            skipped += 1
            continue
        
        # Check if file extension is supported
        file_ext = Path(file_full_path).suffix.lower()
        if file_ext not in supported_extensions:
            print(f"Unsupported format {file_ext} for {uid}")
            skipped += 1
            continue

        # Skip if already processed
        if os.path.exists(os.path.join(output_dir, uid)):
            print(f"Directory already exists for {uid}, skipping.")
            skipped += 1
            continue
        
        os.makedirs(os.path.join(output_dir, uid), exist_ok=True)
        output_prefix = os.path.join(output_dir, uid, uid)
        
        try:
            render_glb(file_full_path, output_prefix)  # render_glb now handles all formats
            processed += 1
        except Exception as e:
            print(f"Error processing {file_full_path}: {str(e)}")
            skipped += 1
    
    print(f"Processing complete: {processed} processed, {skipped} skipped")

# Keep your existing helper functions (get_camera_poses, look_at, is_black_and_white)
# ... [include original functions here] ...

if __name__ == "__main__":
    print("Supported extensions:", get_supported_extensions())
    
    json_path = "/home/benzshawelt/.objaverse/hf-objaverse-v1/object-paths.json"
    output_dir = "./objaverse_images"
    process_objaverse_files(json_path, output_dir)