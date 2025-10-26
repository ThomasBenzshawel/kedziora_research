#!/usr/bin/env python3
"""
Parallel Voxelization Worker
Processes a chunk of GLB files from a JSON list and converts them to voxel tensors
with automatic orientation detection
"""

import json
import argparse
import logging
from pathlib import Path
import sys
import time
import traceback
import gc
import numpy as np
import trimesh
from datetime import datetime
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA

def setup_logging(log_file=None):
    """Set up logging configuration"""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=handlers
    )
    return logging.getLogger(__name__)


def rotation_matrix_from_vectors(vec1, vec2):
    """Find rotation matrix that aligns vec1 to vec2"""
    a = vec1 / np.linalg.norm(vec1)
    b = vec2 / np.linalg.norm(vec2)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    
    if s < 1e-10:  # Vectors are parallel
        if c > 0:
            return np.eye(3)
        else:
            # Find an orthogonal vector
            if abs(a[0]) < 0.9:
                orthogonal = np.array([1, 0, 0])
            else:
                orthogonal = np.array([0, 1, 0])
            v = np.cross(a, orthogonal)
            v = v / np.linalg.norm(v)
            return 2 * np.outer(v, v) - np.eye(3)
    
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


def compute_pca_rotation(vertices):
    """Compute PCA-based rotation matrix"""
    pca = PCA(n_components=3)
    pca.fit(vertices)
    
    # PCA components as rotation matrix
    return pca.components_.T


def find_stable_orientation(mesh, logger=None):
    """
    Find the most stable orientation by testing different rotations
    and selecting the one with lowest center of mass and largest base area
    """
    vertices = mesh.vertices.copy()
    best_score = -np.inf
    best_rotation = np.eye(3)
    
    try:
        # Generate candidate rotations
        candidates = []
        
        # Add principal component alignments
        try:
            pca_rotation = compute_pca_rotation(vertices)
            
            # Generate 24 orientations (6 faces * 4 rotations each)
            for i in range(3):
                for sign1 in [1, -1]:
                    for j in range(3):
                        if i == j:
                            continue
                        for sign2 in [1, -1]:
                            R = np.zeros((3, 3))
                            R[:, 2] = pca_rotation[:, i] * sign1  # Z-axis (up)
                            R[:, 0] = pca_rotation[:, j] * sign2  # X-axis
                            # Complete the right-handed coordinate system
                            R[:, 1] = np.cross(R[:, 2], R[:, 0])
                            candidates.append(R)
        except Exception as e:
            if logger:
                logger.warning(f"PCA rotation failed, using identity: {e}")
            candidates.append(np.eye(3))
        
        # Test each candidate
        for R in candidates:
            rotated_vertices = None
            normalized_vertices = None
            try:
                rotated_vertices = vertices @ R.T
                
                # Normalize to sit on Z=0
                min_z = rotated_vertices[:, 2].min()
                normalized_vertices = rotated_vertices.copy()
                normalized_vertices[:, 2] -= min_z
                
                # Find points near the base (within 10% of height)
                height = normalized_vertices[:, 2].max()
                if height < 1e-10:
                    continue
                    
                base_threshold = 0.1 * height
                base_points = normalized_vertices[normalized_vertices[:, 2] < base_threshold]
                
                if len(base_points) > 3:
                    # Base area (convex hull of base points projected to XY plane)
                    try:
                        base_hull = ConvexHull(base_points[:, :2])
                        base_area = base_hull.volume  # In 2D, volume is area
                        del base_hull
                    except:
                        base_area = 0.01
                    
                    # Center of mass height (lower is better)
                    com_height = normalized_vertices[:, 2].mean() / height
                    
                    # Volume above base (for stability)
                    volume_distribution = np.sum(normalized_vertices[:, 2] < height * 0.3) / len(normalized_vertices)
                    
                    # Score combines large base area, low center of mass, and bottom-heavy distribution
                    score = (base_area * volume_distribution) / (com_height + 0.1)
                    
                    if score > best_score:
                        best_score = score
                        best_rotation = R
            except Exception as e:
                if logger:
                    logger.debug(f"Orientation candidate failed: {e}")
                continue
            finally:
                if rotated_vertices is not None:
                    del rotated_vertices
                if normalized_vertices is not None:
                    del normalized_vertices
        
        return best_rotation
        
    finally:
        del vertices
        gc.collect()


def find_best_base_orientation(mesh, logger=None):
    """
    Find orientation by detecting the best base face of the convex hull
    """
    hull = None
    try:
        # Get convex hull
        hull = ConvexHull(mesh.vertices)
        
        best_score = -np.inf
        best_rotation = np.eye(3)
        
        # Test each face of the convex hull as a potential base
        for simplex in hull.simplices:
            try:
                # Get face vertices
                face_vertices = mesh.vertices[simplex]
                
                # Compute face normal
                v1 = face_vertices[1] - face_vertices[0]
                v2 = face_vertices[2] - face_vertices[0]
                normal = np.cross(v1, v2)
                normal_length = np.linalg.norm(normal)
                
                if normal_length < 1e-10:
                    continue
                    
                normal = normal / normal_length
                
                # Rotate so this normal points down (negative Z)
                rotation = rotation_matrix_from_vectors(normal, np.array([0, 0, -1]))
                rotated_vertices = mesh.vertices @ rotation.T
                
                # Normalize to sit on Z=0
                min_z = rotated_vertices[:, 2].min()
                rotated_vertices[:, 2] -= min_z
                
                height = rotated_vertices[:, 2].max()
                if height < 1e-10:
                    continue
                
                # Compute score based on:
                # 1. Face area (larger is better for base)
                face_area = 0.5 * normal_length
                
                # 2. How many vertices are near this base level
                base_support = np.sum(rotated_vertices[:, 2] < 0.05 * height)
                
                # 3. Center of mass height
                com_height = rotated_vertices[:, 2].mean()
                
                score = (face_area * base_support) / (com_height + 1.0)
                
                if score > best_score:
                    best_score = score
                    best_rotation = rotation
            except Exception as e:
                if logger:
                    logger.debug(f"Hull face evaluation failed: {e}")
                continue
        
        return best_rotation
    except Exception as e:
        if logger:
            logger.warning(f"Convex hull orientation failed: {e}")
        return np.eye(3)
    finally:
        if hull is not None:
            del hull
        gc.collect()


def auto_orient_mesh(mesh, method='stability', logger=None):
    """
    Automatically orient the mesh to have the most likely upright position
    
    Args:
        mesh: Trimesh object
        method: 'stability', 'convex_hull', 'quick', or 'hybrid'
        logger: Logger object for debugging
    
    Returns:
        Oriented mesh (modified in place) and confidence score
    """
    confidence = 0.0
    
    if method == 'stability':
        rotation = find_stable_orientation(mesh, logger)
        confidence = 0.8
    elif method == 'convex_hull':
        rotation = find_best_base_orientation(mesh, logger)
        confidence = 0.7
    elif method == 'hybrid':
        # Try stability first, fall back to convex hull
        rotation = find_stable_orientation(mesh, logger)
        confidence = 0.8
        
        # Validate the result
        mesh_test = mesh.copy()
        try:
            mesh_test.vertices = mesh_test.vertices @ rotation.T
            mesh_test.vertices[:, 2] -= mesh_test.vertices[:, 2].min()
            height = mesh_test.vertices[:, 2].max()
            
            if height > 0:
                com_ratio = mesh_test.vertices[:, 2].mean() / height
                if com_ratio > 0.6:  # Center of mass too high, try convex hull
                    if logger:
                        logger.debug("Stability method resulted in high COM, trying convex hull")
                    rotation = find_best_base_orientation(mesh, logger)
                    confidence = 0.7
        finally:
            del mesh_test
            gc.collect()
    else:
        rotation = np.eye(3)
        confidence = 0.5
    
    # Apply rotation
    mesh.vertices = mesh.vertices @ rotation.T
    
    # Ensure object sits on Z=0
    mesh.vertices[:, 2] -= mesh.vertices[:, 2].min()
    
    return mesh, confidence


def glb_to_voxel_tensor(glb_path, output_path, voxel_resolution=(64, 64, 64),
                        auto_orient=True, orient_method='hybrid', logger=None):
    """
    Convert a GLB file to a binary voxel tensor with optional automatic orientation.
    """
    scene = None
    mesh = None
    voxel_grid = None
    
    try:
        # Load the GLB file
        scene = trimesh.load(glb_path, force='scene')
        
        # Combine all meshes in the scene into a single mesh
        if isinstance(scene, trimesh.Scene):
            meshes = []
            for name, geometry in scene.geometry.items():
                if isinstance(geometry, trimesh.Trimesh):
                    meshes.append(geometry)
            
            if len(meshes) == 0:
                raise ValueError("No valid mesh geometry found in the GLB file")
            
            if len(meshes) == 1:
                mesh = meshes[0]
            else:
                mesh = trimesh.util.concatenate(meshes)
        elif isinstance(scene, trimesh.Trimesh):
            mesh = scene
        else:
            raise ValueError(f"Unexpected type loaded from GLB: {type(scene)}")
        
        # Check if mesh is valid
        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            raise ValueError("Mesh has no vertices or faces")
        
        # Auto-orient the mesh before voxelization
        orientation_confidence = 1.0
        start_time = time.time()
        if auto_orient:
            mesh, orientation_confidence = auto_orient_mesh(mesh, method=orient_method, logger=logger)
            if logger and orientation_confidence < 0.7:
                logger.warning(f"Low orientation confidence ({orientation_confidence:.2f}) for {glb_path}")
        
        if logger:
            elapsed = time.time() - start_time
            logger.debug(f"Auto-orientation took {elapsed:.2f} seconds with confidence {orientation_confidence:.2f}")
        
        # =====================================================================
        # CRITICAL FIX: DO NOT CENTER VERTICALLY!
        # After auto_orient_mesh, the mesh already sits on Z=0
        # We only center in X-Y plane
        # =====================================================================
        
        # Center in X-Y plane only
        xy_center = mesh.vertices[:, :2].mean(axis=0)
        mesh.vertices[:, 0] -= xy_center[0]
        mesh.vertices[:, 1] -= xy_center[1]
        # DO NOT touch Z! mesh.vertices[:, 2] should still have min at 0
        
        # Verify the mesh sits on Z=0 after orientation
        min_z = mesh.vertices[:, 2].min()
        if abs(min_z) > 1e-6:  # Should be very close to 0
            if logger:
                logger.warning(f"Mesh not on Z=0 plane (min_z={min_z:.6f}), adjusting...")
            mesh.vertices[:, 2] -= min_z
        
        # Calculate scale factor to fit the mesh in the voxel grid
        # Leave 1 voxel margin on all sides
        mesh_extents = mesh.extents
        if np.any(mesh_extents == 0):
            raise ValueError("Mesh has zero extent in one or more dimensions")
        
        # Scale to fit within (resolution - 2) to leave 1 voxel border
        max_extent = max(mesh_extents)
        max_grid = min(voxel_resolution) - 2  # Leave margin
        scale_factor = max_grid / max_extent
        
        # Scale the mesh uniformly
        mesh.vertices *= scale_factor
        
        # Calculate voxel pitch (size of each voxel in mesh units)
        # After scaling, the mesh fits in roughly (resolution-2) voxels
        # So pitch should be approximately max_extent_after_scaling / (resolution-2)
        pitch = mesh.extents.max() / (min(voxel_resolution) - 2)
        
        # Perform voxelization
        voxel_grid = mesh.voxelized(pitch=pitch)
        
        # Fill the interior (comment this out if you want shell-only)
        voxel_grid = voxel_grid.fill()
        
        # Convert to binary numpy array
        voxel_matrix = voxel_grid.matrix
        
        # =====================================================================
        # CRITICAL FIX: ALIGN BOTTOM OF OBJECT TO Z=0 OF GRID
        # Do NOT center in Z dimension!
        # =====================================================================
        
        binary_tensor = np.zeros(voxel_resolution, dtype=np.uint8)
        
        # Calculate placement in grid
        src_shape = voxel_matrix.shape
        
        # For X and Y: center in grid
        x_offset = (voxel_resolution[0] - src_shape[0]) // 2
        y_offset = (voxel_resolution[1] - src_shape[1]) // 2
        
        # For Z: START AT BOTTOM (Z=0), DO NOT CENTER!
        z_offset = 0  # Always start at bottom
        
        # Calculate safe bounds (handle case where voxel_matrix is larger than grid)
        x_start_dst = max(0, x_offset)
        y_start_dst = max(0, y_offset)
        z_start_dst = 0  # Always start at bottom
        
        x_start_src = max(0, -x_offset)
        y_start_src = max(0, -y_offset)
        z_start_src = 0
        
        x_end_dst = min(voxel_resolution[0], x_offset + src_shape[0])
        y_end_dst = min(voxel_resolution[1], y_offset + src_shape[1])
        z_end_dst = min(voxel_resolution[2], src_shape[2])
        
        x_end_src = x_start_src + (x_end_dst - x_start_dst)
        y_end_src = y_start_src + (y_end_dst - y_start_dst)
        z_end_src = z_start_src + (z_end_dst - z_start_dst)
        
        # Copy voxels into target tensor
        binary_tensor[x_start_dst:x_end_dst, 
                     y_start_dst:y_end_dst, 
                     z_start_dst:z_end_dst] = \
            voxel_matrix[x_start_src:x_end_src,
                        y_start_src:y_end_src,
                        z_start_src:z_end_src].astype(np.uint8)
        
        # =====================================================================
        # VALIDATION: Ensure bottom-up structure is correct
        # =====================================================================
        
        # Check that bottom layer has voxels
        bottom_layer_voxels = binary_tensor[:, :, 0].sum()
        if bottom_layer_voxels == 0:
            raise ValueError(f"No voxels in bottom layer (Z=0) for {glb_path}! "
                           f"Object may be floating or orientation failed.")
        
        # Check that object is grounded (has voxels in bottom 10% of layers)
        bottom_10_percent = max(1, voxel_resolution[2] // 10)
        bottom_region_voxels = binary_tensor[:, :, :bottom_10_percent].sum()
        if bottom_region_voxels == 0:
            raise ValueError(f"No voxels in bottom {bottom_10_percent} layers! "
                           f"Object not properly grounded.")
        
        # Log statistics
        total_voxels = binary_tensor.sum()
        if logger:
            logger.debug(f"Voxel stats - Total: {total_voxels}, "
                        f"Bottom layer: {bottom_layer_voxels}, "
                        f"Bottom 10%: {bottom_region_voxels}")
        
        # Save to file
        if logger:
            logger.debug(f"Saving voxel tensor to {output_path}")
        np.save(output_path, binary_tensor)
        
        # Save metadata for training
        metadata = {
            'orientation_confidence': float(orientation_confidence),
            'scale_factor': float(scale_factor),
            'original_extents': mesh_extents.tolist(),
            'voxel_pitch': float(pitch),
            'occupied_voxels': int(total_voxels),
            'bottom_layer_voxels': int(bottom_layer_voxels)
        }
        metadata_path = str(output_path).replace('.npy', '_meta.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return binary_tensor, orientation_confidence
        
    finally:
        if voxel_grid is not None:
            del voxel_grid
        if scene is not None:
            del scene
        if mesh is not None:
            del mesh
        gc.collect()


def process_chunk(json_path, scan_dir, scan_dir_2, output_dir, check_dir, chunk_id, total_chunks, 
                  auto_orient, orient_method, voxel_resolution, logger):
    """
    Process a chunk of the GLB files
    """

    # turn voxel_resolution into a tuple if it's an int
    if isinstance(voxel_resolution, int):
        voxel_resolution = (voxel_resolution, voxel_resolution, voxel_resolution)
    
    # Scan over all provided scan directories and find GLB files to process (ignore json for now)
    items = []
    for base_dir in [scan_dir, scan_dir_2]:
        base_path = Path(base_dir)
        if not base_path.exists():
            logger.warning(f"Scan directory does not exist: {base_path}")
            continue
        
        for glb_path in base_path.rglob('*.glb'):
            try:
                # Create a unique ID based on relative path for output naming
                uid = glb_path.stem
                
                # Store absolute path instead of relative path
                items.append((uid, glb_path))
            except Exception as e:
                logger.error(f"Failed to process path {glb_path}: {e}")
                continue

    if len(items) == 0:
        raise ValueError("No GLB files found in the provided scan directories")
    
    items = sorted(items, key=lambda x: x[0])  # Sort by UID for consistency
    total_items = len(items)
    
    logger.info(f"Total items: {total_items}")
    
    # Calculate chunk boundaries
    chunk_size = total_items // total_chunks
    remainder = total_items % total_chunks
    
    # Distribute remainder across first chunks
    if chunk_id < remainder:
        start_idx = chunk_id * (chunk_size + 1)
        end_idx = start_idx + chunk_size + 1
    else:
        start_idx = chunk_id * chunk_size + remainder
        end_idx = start_idx + chunk_size
    
    chunk_items = items[start_idx:end_idx]
    logger.info(f"Worker {chunk_id}: Processing items {start_idx} to {end_idx-1} ({len(chunk_items)} items)")
    if auto_orient:
        logger.info(f"Auto-orientation enabled with method: {orient_method}")
    
    # Statistics
    processed = 0
    skipped = 0
    failed = 0
    low_confidence_count = 0
    start_time = time.time()
    
    for idx, (uid, glb_path) in enumerate(chunk_items):
        binary_tensor = None
        try:
            # glb_path is now already absolute, no need to construct it
            
            # Maintain directory structure for output
            # Extract the directory structure from the relative path
            output_subdir = Path(output_dir) / str(voxel_resolution[0])
            output_subdir.mkdir(parents=True, exist_ok=True)
            
            # Output file path (change extension from .glb to .npy)
            output_filename = uid + '.npy'
            output_path = output_subdir / output_filename
            
            # Check if already processed
            if check_dir:
                check_path = Path(check_dir) / output_filename
            else:
                check_path = output_path
            
            if check_path.exists():
                skipped += 1
                if (idx + 1) % 100 == 0:
                    logger.info(f"Progress: {idx+1}/{len(chunk_items)} - Skipped (already exists): {uid}")
                continue
            
            # Check if input exists
            if not glb_path.exists():
                logger.warning(f"Input file not found: {glb_path}")
                failed += 1
                continue
            
            # Process the file
            if (idx + 1) % 10 == 0:
                elapsed = time.time() - start_time
                rate = (processed + skipped) / elapsed if elapsed > 0 else 0
                eta = (len(chunk_items) - idx - 1) / rate if rate > 0 else 0
                logger.info(f"Progress: {idx+1}/{len(chunk_items)} - Processing: {uid} "
                          f"(Rate: {rate:.2f} items/s, ETA: {eta/60:.1f} min)")
            
            # Convert GLB to voxel tensor with orientation
            binary_tensor, orientation_confidence = glb_to_voxel_tensor(
                glb_path, output_path, voxel_resolution=voxel_resolution,
                auto_orient=auto_orient, orient_method=orient_method, logger=logger
            )
            # Track low confidence orientations
            if orientation_confidence < 0.7:
                low_confidence_count += 1
            
            # Quick validation
            occupied_voxels = np.sum(binary_tensor)
            if occupied_voxels == 0:
                logger.warning(f"Warning: Empty voxel tensor for {uid}")
            
            processed += 1
            
        except Exception as e:
            failed += 1
            logger.error(f"Failed to process {uid}: {str(e)}")
            if logger.level == logging.DEBUG:
                logger.debug(traceback.format_exc())
            continue
        finally:
            if binary_tensor is not None:
                del binary_tensor
            
            # Periodic garbage collection
            if (idx + 1) % 10 == 0:
                gc.collect()
    
    # Final statistics
    elapsed_time = time.time() - start_time
    logger.info(f"Worker {chunk_id} completed in {elapsed_time/60:.2f} minutes")
    logger.info(f"Processed: {processed}, Skipped: {skipped}, Failed: {failed}")
    if auto_orient and processed > 0:
        logger.info(f"Low confidence orientations: {low_confidence_count}/{processed} ({100*low_confidence_count/processed:.1f}%)")
    logger.info(f"Average rate: {(processed + skipped)/elapsed_time:.2f} items/second")
    
    return processed, skipped, failed

def main():
    parser = argparse.ArgumentParser(description='Voxelization worker for parallel processing with auto-orientation')
    
    parser.add_argument('--json_path', required=False, help='Path to JSON file with GLB paths', default="/home/ad.msoe.edu/benzshawelt/.objaverse/hf-objaverse-v1/object-paths.json")
    parser.add_argument('--scan_dir', required=False, help='Base directory for GLB files', default="/home/ad.msoe.edu/benzshawelt/.objaverse")
    parser.add_argument('--scan_dir_2', required=False, help='Secondary directory for GLB files', default="/home/ad.msoe.edu/benzshawelt/objaverse_temp")
    parser.add_argument('--output_dir', required=False, help='Output directory for voxel tensors', default="../objaverse_voxels")
    parser.add_argument('--chunk_id', type=int, required=False, help='Worker chunk ID', default=0)
    parser.add_argument('--total_chunks', type=int, required=False, help='Total number of chunks', default=1)
    parser.add_argument('--check_dir', help='Directory to check for existing files (default: output_dir)', default="../objaverse_voxels")
    parser.add_argument('--log_file', help='Log file path')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--voxel_resolution', type=int, default=64, help='Voxel grid resolution (default: 64)')
    
    # Auto-orientation options
    parser.add_argument('--no_auto_orient', action='store_true', 
                        help='Disable automatic orientation (use original orientation)')
    parser.add_argument('--orient_method', choices=['stability', 'convex_hull', 'hybrid'],
                        default='hybrid',
                        help='Orientation method: stability (most robust), convex_hull (geometric), '
                             'or hybrid (stability with convex_hull fallback)')

    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging(args.log_file)
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    logger.info("="*60)
    logger.info(f"Voxelization Worker {args.chunk_id}/{args.total_chunks}")
    logger.info(f"Resolution: {args.voxel_resolution}^3")
    logger.info(f"Auto-orientation: {'Disabled' if args.no_auto_orient else f'Enabled ({args.orient_method} method)'}")
    logger.info(f"Started at: {datetime.now()}")
    logger.info("="*60)
    
    try:
        # Process the chunk
        processed, skipped, failed = process_chunk(
            args.json_path,
            args.scan_dir,
            args.scan_dir_2,
            args.output_dir,
            args.check_dir,
            args.chunk_id,
            args.total_chunks,
            not args.no_auto_orient,  # auto_orient flag
            args.orient_method,
            args.voxel_resolution,
            logger
        )
        
        # Exit with error code if all files failed
        if processed == 0 and skipped == 0:
            logger.error("No files were successfully processed")
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"Worker failed with error: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)
    
    logger.info(f"Worker {args.chunk_id} finished successfully")


if __name__ == "__main__":
    main()