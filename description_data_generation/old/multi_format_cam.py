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

def clean_mesh(mesh):
    """Clean individual mesh to prevent numerical issues"""
    try:
        if mesh is None:
            return None
            
        # Check if mesh has required attributes
        if not hasattr(mesh, 'vertices'):
            print("Mesh has no vertices attribute")
            return None
            
        # Check if vertices is None or empty
        if mesh.vertices is None or len(mesh.vertices) == 0:
            print("Empty mesh - no vertices")
            return None
            
        # Check if mesh has faces (for meshes that should have them)
        if hasattr(mesh, 'faces'):
            if mesh.faces is None:
                print("Mesh faces is None")
                return None
            elif len(mesh.faces) == 0:
                print("Empty mesh - no faces")
                return None
        
        # Try mesh cleaning operations with individual error handling
        try:
            if hasattr(mesh, 'remove_duplicate_faces'):
                mesh.update_faces(mesh.unique_faces())
        except Exception as e:
            print(f"Warning: Could not remove duplicate faces: {e}")
            
        try:
            if hasattr(mesh, 'remove_degenerate_faces'):
                mesh.update_faces(mesh.nondegenerate_faces(height=1e-10))
        except Exception as e:
            print(f"Warning: Could not remove degenerate faces: {e}")
            
        try:
            if hasattr(mesh, 'remove_unreferenced_vertices'):
                mesh.remove_unreferenced_vertices()
        except Exception as e:
            print(f"Warning: Could not remove unreferenced vertices: {e}")
        try:
            if hasattr(mesh, 'fix_normals'):
                mesh.fix_normals()
        except Exception as e:
            print(f"Warning: Could not fix normals: {e}")
        
        # Re-check vertices after cleaning
        if mesh.vertices is None or len(mesh.vertices) == 0:
            print("Mesh became empty after cleaning")
            return None
        
        # Check bounds and scale if necessary with safe operations
        try:
            if hasattr(mesh, 'bounds') and mesh.bounds is not None:
                bounds = mesh.bounds
                if bounds is not None and len(bounds) == 2 and bounds[0] is not None and bounds[1] is not None:
                    size = np.linalg.norm(bounds[1] - bounds[0])
                    
                    # Handle extremely small meshes
                    if size < 1e-10:
                        print("Mesh too small, skipping")
                        return None
                        
                    # Handle extremely large meshes
                    if size > 1e6:
                        print(f"Large mesh detected (size: {size}), scaling down")
                        scale_factor = 1.0 / size
                        try:
                            mesh.apply_scale(scale_factor)
                        except Exception as e:
                            print(f"Warning: Could not scale mesh: {e}")
        except Exception as e:
            print(f"Warning: Could not check mesh bounds: {e}")
        
        # Final verification
        if not hasattr(mesh, 'vertices') or mesh.vertices is None or len(mesh.vertices) == 0:
            return None
            
        return mesh
        
    except Exception as e:
        print(f"Error cleaning mesh: {e}")
        return None

def clean_and_validate_scene(scene):
    """Clean and validate scene geometry to prevent eigenvalue convergence issues"""
    try:
        # Handle different scene types
        if hasattr(scene, 'geometry'):
            # Multi-geometry scene
            cleaned_geometries = {}
            for name, geom in scene.geometry.items():
                cleaned_geom = clean_mesh(geom)
                if cleaned_geom is not None:
                    cleaned_geometries[name] = cleaned_geom
            
            if not cleaned_geometries:
                return None
                
            # Create new scene with cleaned geometries
            scene.geometry = cleaned_geometries
            
        elif hasattr(scene, 'vertices'):
            # Single mesh
            scene = clean_mesh(scene)

        if scene is None or not hasattr(scene, 'geometry') or len(scene.geometry) == 0 or \
              (hasattr(scene, 'vertices') and (scene.vertices is None or len(scene.vertices) == 0)):
            print("Scene became empty after cleaning")
            return None
            
        return scene
        
    except Exception as e:
        print(f"Error during scene cleaning: {e}")
        return None

def safe_scene_properties(scene):
    """Safely compute scene properties with fallbacks"""
    try:
        # Try to get centroid with None check
        centroid = None
        if hasattr(scene, 'centroid'):
            try:
                centroid = scene.centroid
                if centroid is None or not isinstance(centroid, np.ndarray):
                    centroid = None
            except:
                centroid = None
        
        if centroid is None:
            # Fallback: compute centroid manually
            if hasattr(scene, 'vertices') and scene.vertices is not None and len(scene.vertices) > 0:
                centroid = np.mean(scene.vertices, axis=0)
            elif hasattr(scene, 'geometry') and scene.geometry:
                # Try to get vertices from first geometry
                for geom in scene.geometry.values():
                    if hasattr(geom, 'vertices') and geom.vertices is not None and len(geom.vertices) > 0:
                        centroid = np.mean(geom.vertices, axis=0)
                        break
            
            if centroid is None:
                centroid = np.array([0, 0, 0])
        
        # Try to get bounds with extensive None checking
        bounds = None
        if hasattr(scene, 'bounds'):
            try:
                bounds = scene.bounds
                if bounds is None or len(bounds) != 2:
                    bounds = None
                elif bounds[0] is None or bounds[1] is None:
                    bounds = None
                elif not isinstance(bounds[0], np.ndarray) or not isinstance(bounds[1], np.ndarray):
                    bounds = None
            except:
                bounds = None
        
        if bounds is None:
            # Fallback: compute bounds manually
            if hasattr(scene, 'vertices') and scene.vertices is not None and len(scene.vertices) > 0:
                bounds = np.array([np.min(scene.vertices, axis=0), 
                                 np.max(scene.vertices, axis=0)])
            elif hasattr(scene, 'geometry') and scene.geometry:
                # Try to get bounds from geometries
                all_vertices = []
                for geom in scene.geometry.values():
                    if hasattr(geom, 'vertices') and geom.vertices is not None and len(geom.vertices) > 0:
                        all_vertices.append(geom.vertices)
                
                if all_vertices:
                    combined_vertices = np.vstack(all_vertices)
                    bounds = np.array([np.min(combined_vertices, axis=0), 
                                     np.max(combined_vertices, axis=0)])
            
            if bounds is None:
                bounds = np.array([[-1, -1, -1], [1, 1, 1]])
        
        # Validate bounds
        if bounds[0] is None or bounds[1] is None:
            bounds = np.array([[-1, -1, -1], [1, 1, 1]])
        
        # Ensure bounds make sense (min < max)
        if np.any(bounds[1] <= bounds[0]):
            print("Warning: Invalid bounds detected, using defaults")
            bounds = np.array([[-1, -1, -1], [1, 1, 1]])
        
        return centroid, bounds
        
    except Exception as e:
        print(f"Error computing scene properties: {e}")
        # Return default values
        return np.array([0, 0, 0]), np.array([[-1, -1, -1], [1, 1, 1]])

def normalize_scene_orientation_safe(scene, bounds):
    """Safely normalize scene orientation"""
    try:
        # Validate inputs
        if scene is None:
            return scene
        if bounds is None or len(bounds) != 2:
            return scene
        if bounds[0] is None or bounds[1] is None:
            return scene
            
        # Safely compute size
        try:
            size = np.linalg.norm(bounds[1] - bounds[0])
        except (TypeError, ValueError):
            print("Warning: Could not compute scene size")
            return scene
        
        if not np.isfinite(size) or size <= 0:
            print("Warning: Invalid scene size")
            return scene
        
        # Scale if too small or too large, but be conservative
        scale_factor = None
        if size < 0.001:
            scale_factor = 10.0
            print(f"Scene very small ({size}), scaling up by {scale_factor}")
        elif size > 1000:
            scale_factor = 1.0 / (size / 10.0)
            print(f"Scene very large ({size}), scaling down by {scale_factor}")
        
        if scale_factor is not None:
            try:
                if hasattr(scene, 'apply_scale'):
                    scene.apply_scale(scale_factor)
                elif hasattr(scene, 'geometry'):
                    for geom in scene.geometry.values():
                        if hasattr(geom, 'apply_scale'):
                            geom.apply_scale(scale_factor)
            except Exception as e:
                print(f"Warning: Could not apply scale to scene: {e}")
                
        return scene
        
    except Exception as e:
        print(f"Warning: Could not normalize scene orientation: {e}")
        return scene

def look_at_safe(eye, target):
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
        print(f"Error in look_at: {e}")
        return np.eye(3)

def get_camera_poses_safe(centroid, bounds):
    """Safely compute camera poses with fallbacks"""
    try:
        # Validate inputs
        if centroid is None:
            centroid = np.array([0, 0, 0])
        if bounds is None or len(bounds) != 2:
            bounds = np.array([[-1, -1, -1], [1, 1, 1]])
        if bounds[0] is None or bounds[1] is None:
            bounds = np.array([[-1, -1, -1], [1, 1, 1]])
            
        # Safely compute diagonal
        try:
            diagonal = np.linalg.norm(bounds[1] - bounds[0])
        except (TypeError, IndexError, ValueError):
            diagonal = 2.0  # Default diagonal
        
        # Ensure diagonal is reasonable
        if diagonal <= 0 or not np.isfinite(diagonal):
            diagonal = 2.0
        elif diagonal < 0.1:
            diagonal = 1.0
        elif diagonal > 1000:
            diagonal = 10.0
        
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
            try:
                pose[:3, 3] = centroid + np.array(pos)
                pose[:3, :3] = look_at_safe(pose[:3, 3], centroid)
            except Exception as e:
                print(f"Warning: Error setting camera pose: {e}")
                # Fallback to identity with offset
                pose[:3, 3] = np.array(pos)
                pose[:3, :3] = np.eye(3)
            camera_poses.append(pose)
        
        return camera_poses, diagonal
        
    except Exception as e:
        print(f"Error computing camera poses: {e}")
        # Return default poses
        default_poses = []
        for i in range(6):
            pose = np.eye(4)
            pose[2, 3] = 2.0  # Move camera back
            default_poses.append(pose)
        return default_poses, 2.0

def is_black_and_white_improved(image, dark_threshold=40, bright_threshold=200, saturation_threshold=30):
    """
    Improved black/white detection with simpler logic
    
    Args:
        image: RGB image array
        dark_threshold: Value below which image is considered too dark (0-255)
        bright_threshold: Value above which image is considered too bright (0-255) 
        saturation_threshold: Saturation below which image lacks color (0-255)
    
    Returns:
        'black', 'white', or 'color'
    """
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Get mean values
    val_mean = np.mean(hsv[:, :, 2])  # Brightness/Value
    sat_mean = np.mean(hsv[:, :, 1])  # Saturation
    
    # Check if image has sufficient color saturation
    if sat_mean > saturation_threshold:
        return "color"
    
    # If low saturation, classify by brightness
    if val_mean < dark_threshold:
        return "black"
    elif val_mean > bright_threshold:
        return "white"
    else:
        # Grayscale but acceptable - treat as color
        return "color"

def adaptive_lighting_adjustment(current_intensity, image_state, iteration):
    """
    More conservative lighting adjustment
    """
    if image_state == "white":
        # Reduce intensity more conservatively for overexposed images
        multiplier = 0.6 if iteration < 5 else 0.4
        return current_intensity * multiplier
    elif image_state == "black":
        # Increase intensity more conservatively for underexposed images  
        multiplier = 1.4 if iteration < 5 else 1.8
        return current_intensity * multiplier
    else:
        return current_intensity

def render_glb_safe(file_path, output_prefix, resolution=(800, 600)):
    """Enhanced render function with robust error handling"""
    try:

        # import trimesh
        # scene = trimesh.load(file_path)
        
        # if scene.geometry:
        #     print(f"Found {len(scene.geometry)} geometry objects")
            
        #     # Access individual meshes
        #     for name, geometry in scene.geometry.items():
        #         print(f"Geometry '{name}': {len(geometry.vertices)} vertices")


        # Load scene using the multi-format loader
        scene = MultiFormatLoader.load_scene(file_path)
        
        if scene is None:
            raise ValueError(f"Could not load scene from {file_path}")
        
        scene = clean_and_validate_scene(scene)
        
        if scene is None:
            print(f"Scene became invalid after cleaning: {file_path}, reloading original scene.")
            # reload the original scene and try to use it
            scene = MultiFormatLoader.load_scene(file_path)
            if scene is None:
                raise ValueError(f"Could not load original scene from {file_path}")

        # Safely compute scene properties
        centroid, bounds = safe_scene_properties(scene)
        
        # Handle coordinate system differences with safe operations
        scene = normalize_scene_orientation_safe(scene, centroid, bounds)
        
        # Convert to pyrender scene with error handling
        try:
            pyrender_scene = pyrender.Scene.from_trimesh_scene(
                scene, bg_color=[0.6, 0.6, 0.6], ambient_light=[1, 1, 1]
            )
        except Exception as e:
            print(f"Error converting to pyrender scene: {e}")
            # Try with minimal scene
            pyrender_scene = pyrender.Scene(bg_color=[0.6, 0.6, 0.6], ambient_light=[1, 1, 1])
            
            # Manually add geometries
            if hasattr(scene, 'geometry'):
                for name, geom in scene.geometry.items():
                    try:
                        mesh = pyrender.Mesh.from_trimesh(geom)
                        pyrender_scene.add(mesh)
                    except:
                        continue
            elif hasattr(scene, 'vertices'):
                try:
                    mesh = pyrender.Mesh.from_trimesh(scene)
                    pyrender_scene.add(mesh)
                except:
                    pass
        
        # Safely compute camera poses
        camera_poses, scene_diagonal = get_camera_poses_safe(centroid, bounds)
        
        # Fixed FOV calculation with bounds checking
        camera_distance = scene_diagonal * 1.15 # add a 15% margin
        camera_distance = max(camera_distance, 0.1)  # Prevent division by zero
        camera_distance = min(camera_distance, 300.0)  # Cap maximum distance
        fov = min(2 * np.arctan(scene_diagonal / (2 * camera_distance)), np.pi/2)  # Cap FOV

        perspectives = ['front', 'back', 'right', 'left', 'up', 'down']
        
        # add a "sun" point light to the scene
        pyrender_scene.add(pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=0.5), pose=np.eye(4))
        # add another "sun" point light to the scene in the opposite direction
        pyrender_scene.add(pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=0.5), pose=np.array([[1, 0, 0, 0],
                                                                                  [0, 1, 0, 0],
                                                                                  [0, 0, 1, -2 * camera_distance],
                                                                                  [0, 0, 0, 1]]))

        camera = pyrender.PerspectiveCamera(yfov=fov)
        # Create renderer once
        r = pyrender.OffscreenRenderer(viewport_width=resolution[0], viewport_height=resolution[1])

        # add camera poses
        for i, pose in enumerate(camera_poses):
            light_intensity = 2.0  # Start with reasonable intensity
            
            # Create initial light
            light = pyrender.DirectionalLight(
                color=np.ones(3), 
                intensity=light_intensity,
            )
            light_node = pyrender_scene.add(light, pose=pose)
            
            best_image = None
            for iteration in range(10):
                # Render image
                camera_node = pyrender_scene.add(camera, pose=pose)
                color, _ = r.render(pyrender_scene)
                pyrender_scene.remove_node(camera_node)
                
                # Check image quality
                image_state = is_black_and_white_improved(color)
                
                if image_state == "color":
                    # Good image - save and break
                    best_image = color
                    break
                
                # Store first acceptable image as fallback
                if best_image is None or (image_state != "black" and image_state != "white"):
                    best_image = color
                
                # Adjust lighting if not final iteration
                if iteration < 9:
                    new_intensity = adaptive_lighting_adjustment(light_intensity, image_state, iteration)
                    
                    # Prevent extreme values
                    new_intensity = np.clip(new_intensity, 0.1, 50.0)
                    
                    if abs(new_intensity - light_intensity) < 0.01:
                        # No significant change, stop adjusting
                        break
                    
                    light_intensity = new_intensity
                    
                    # Update light
                    pyrender_scene.remove_node(light_node)
                    light = pyrender.SpotLight(
                        color=np.ones(3), 
                        intensity=light_intensity,
                        innerConeAngle=np.pi/8.0, 
                        outerConeAngle=np.pi/4.0
                    )
                    light_node = pyrender_scene.add(light, pose=pose)
                    
                    print(f"Iteration {iteration+1}: {image_state} image, adjusting intensity to {light_intensity:.2f}")
                else:
                    print(f"Warning: {perspectives[i]} view remained {image_state} after 10 iterations.")
            
            # Save the best image we found
            if best_image is not None:
                img = Image.fromarray(best_image)
                output_path = f"{output_prefix}_{perspectives[i]}.jpg"
                img.save(output_path, 'JPEG', quality=95)
            
        # Clean up
        pyrender_scene.remove_node(light_node)
            
    except Exception as e:
        print(f"Failed to render {file_path}: {e}")
        raise
    finally:
        r.delete()

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

def get_supported_extensions():
    """Return list of supported file extensions"""
    base_extensions = ['.obj', '.glb', '.gltf', '.stl', '.dae', '.ply']
    
    if HAS_BLENDER:
        base_extensions.extend(['.fbx', '.abc', '.blend'])
    
    if HAS_USD:
        base_extensions.extend(['.usd', '.usda', '.usdz'])
    
    return base_extensions

def process_directory_files_safe(root_dir, output_dir, use_folder_name_as_uid=True):
    """Safe version of directory processing with enhanced error handling"""
    object_paths = scan_directory_for_3d_files(root_dir, use_folder_name_as_uid)
    
    if not object_paths:
        print("No 3D files found in the specified directory!")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    processed = 0
    skipped = 0
    
    for uid, relative_file_path in tqdm(object_paths.items(), desc="Processing files"):
        file_full_path = os.path.join(root_dir, relative_file_path)
        
        if not os.path.exists(file_full_path):
            print(f"File no longer exists: {file_full_path}")
            skipped += 1
            continue

        output_uid_dir = os.path.join(output_dir, uid)
        if os.path.exists(output_uid_dir):
            print(f"Directory already exists for {uid}, skipping.")
            skipped += 1
            continue
        
        os.makedirs(output_uid_dir, exist_ok=True)
        output_prefix = os.path.join(output_uid_dir, uid)
        
        try:
            print(f"Processing: {file_full_path}")
            render_glb_safe(file_full_path, output_prefix)  # Use safe version
            processed += 1
        except Exception as e:
            print(f"Error processing {file_full_path}: {str(e)}")
            # Clean up partial output directory
            if os.path.exists(output_uid_dir) and not os.listdir(output_uid_dir):
                os.rmdir(output_uid_dir)
            skipped += 1
    
    print(f"Processing complete: {processed} processed, {skipped} skipped")

def process_objaverse_files_safe(json_path, output_dir):
    """Updated JSON-based processing with safety improvements"""
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
            render_glb_safe(file_full_path, output_prefix)  # Use safe version
            processed += 1
        except Exception as e:
            print(f"Error processing {file_full_path}: {str(e)}")
            # Clean up partial output directory
            output_uid_dir = os.path.join(output_dir, uid)
            if os.path.exists(output_uid_dir) and not os.listdir(output_uid_dir):
                os.rmdir(output_uid_dir)
            skipped += 1
    
    print(f"Processing complete: {processed} processed, {skipped} skipped")

if __name__ == "__main__":
    print("Supported extensions:", get_supported_extensions())
    
    # Directory-based processing
    root_directory = "./objaverse_xl/hf-objaverse-v1" # ./objaverse_xl/hf-objaverse-v1
    output_dir = "./objaverse_images"
    
    # Process directory with safe functions
    process_directory_files_safe(root_directory, output_dir, use_folder_name_as_uid=True)
    
    # Alternatively, for JSON-based processing:
    # json_file_path = "./object_paths.json"
    # process_objaverse_files_safe(json_file_path, output_dir)