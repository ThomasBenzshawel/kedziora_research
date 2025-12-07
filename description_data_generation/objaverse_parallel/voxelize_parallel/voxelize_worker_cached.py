#!/usr/bin/env python3
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
import multiprocessing as mp


def setup_logging():
    """Set up logging configuration"""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler(sys.stdout)]

    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=handlers
    )
    return logging.getLogger(__name__)

def rotation_matrix_from_vectors(vec1, vec2):
    """Find rotation matrix that aligns vec1 to vec2"""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    # Handle zero-length vectors
    if norm1 < 1e-10 or norm2 < 1e-10:
        return np.eye(3)
    
    a = vec1 / norm1
    b = vec2 / norm2
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    
    if s < 1e-10:  # Vectors are parallel
        if c > 0:
            return np.eye(3)
        else:
            # 180-degree rotation
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
    and selecting the one with the largest base area and lowest center of mass.
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
        del candidates
        gc.collect()


def auto_orient_mesh(mesh, logger=None):
    """
    Automatically orient the mesh so its most stable surface sits on Z=0.
    """
    rotation = find_stable_orientation(mesh, logger)
    confidence = 0.85  # High confidence for stability method
    
    # Apply rotation
    mesh.apply_transform(np.vstack([
        np.hstack([rotation, [[0], [0], [0]]]),
        [0, 0, 0, 1]
    ]))
    
    # After rotation, ensure object sits on Z=0
    min_z = mesh.vertices[:, 2].min()
    mesh.vertices[:, 2] -= min_z
    z_adjustment = min_z
    
    return mesh, confidence, rotation, z_adjustment


def voxelize_mesh(mesh, voxel_resolution):
    """Convert mesh to binary voxel grid"""
    try:
        # Handle tuple input (grids are always cubic)
        if isinstance(voxel_resolution, tuple):
            voxel_resolution = voxel_resolution[0]
        
        # Validate mesh has non-zero extents
        max_extent = mesh.extents.max()
        if max_extent < 1e-6:
            raise ValueError(f"Degenerate mesh with extent {max_extent}")
        
        # Calculate pitch with safety check
        pitch = max_extent / voxel_resolution
        if pitch <= 0 or not np.isfinite(pitch):
            raise ValueError(f"Invalid pitch: {pitch}")
        
        # Use trimesh's voxelize with validated pitch
        voxelized = mesh.voxelized(pitch=pitch)
        
        # Convert to dense boolean array
        voxel_grid = voxelized.matrix
        del voxelized  # Free memory immediately
        gc.collect()
        
        # Ensure cubic grid
        current_shape = voxel_grid.shape
        max_dim = max(current_shape)
        
        if max_dim > voxel_resolution:
            # Downsample if too large - use chunked processing for memory
            from scipy.ndimage import zoom
            scale_factors = [voxel_resolution / d for d in current_shape]
            
            # Free original before creating scaled version
            voxel_grid_scaled = zoom(voxel_grid.astype(np.float32), scale_factors, order=0) > 0.5
            del voxel_grid
            voxel_grid = voxel_grid_scaled
            del voxel_grid_scaled
            gc.collect()
        
        # Pad to cubic
        padded_grid = np.zeros((voxel_resolution, voxel_resolution, voxel_resolution), dtype=bool)
        
        # Center the object in the grid
        sx, sy, sz = voxel_grid.shape
        x_start = (voxel_resolution - sx) // 2
        y_start = (voxel_resolution - sy) // 2
        z_start = 0  # Bottom-aligned
        
        # Ensure we don't exceed bounds
        x_end = min(x_start + sx, voxel_resolution)
        y_end = min(y_start + sy, voxel_resolution)
        z_end = min(z_start + sz, voxel_resolution)
        
        padded_grid[x_start:x_end, y_start:y_end, z_start:z_end] = \
            voxel_grid[:x_end-x_start, :y_end-y_start, :z_end-z_start]
        
        del voxel_grid
        gc.collect()
        
        return padded_grid.astype(np.uint8)
        
    except Exception as e:
        raise ValueError(f"Voxelization failed: {str(e)}")
    finally:
        gc.collect()


def validate_voxel_tensor(voxel_tensor, logger=None):
    """Validate the voxel tensor for training compatibility"""
    warnings = []
    
    # Check if cubic
    if len(set(voxel_tensor.shape)) != 1:
        warnings.append(f"Non-cubic grid: {voxel_tensor.shape}")
    
    # Check if not empty
    occupied = np.sum(voxel_tensor)
    if occupied == 0:
        warnings.append("Empty voxel grid")
    
    # Check if bottom-aligned (should have voxels near Z=0)
    bottom_slice = voxel_tensor[:, :, :5]
    if np.sum(bottom_slice) == 0:
        warnings.append("No voxels near bottom (Z=0)")
    
    # Check if too sparse or too dense
    fill_ratio = occupied / voxel_tensor.size
    if fill_ratio < 0.001:
        warnings.append(f"Very sparse: {fill_ratio:.4%} filled")
    elif fill_ratio > 0.9:
        warnings.append(f"Very dense: {fill_ratio:.4%} filled")
    
    if warnings and logger:
        logger.warning(f"Validation warnings: {', '.join(warnings)}")
    
    return warnings


def glb_to_voxel_tensor(glb_path, output_path, metadata_path, voxel_resolution=64, 
                        auto_orient=True, logger=None):
    """
    Convert a GLB file to a binary voxel tensor with metadata.
    """
    mesh = None

    if isinstance(voxel_resolution, int):
        voxel_resolution = (voxel_resolution, voxel_resolution, voxel_resolution)
    
    try:
        # Load mesh
        mesh = trimesh.load(glb_path, force='mesh')
        
        if not isinstance(mesh, trimesh.Trimesh):
            if isinstance(mesh, trimesh.Scene):
                scene = mesh
                mesh = scene.dump(concatenate=True)
                del scene
                gc.collect()
            else:
                raise ValueError(f"Unsupported mesh type: {type(mesh)}")
        
        # NOW clean the mesh (after we know it's a Trimesh)
        mask = mesh.unique_faces() & mesh.nondegenerate_faces(height=1e-8)
        mesh.update_faces(mask)
        
        mesh.remove_unreferenced_vertices()
        mesh.merge_vertices()
        mesh.fix_normals()
        
        # Validate mesh before processing
        if len(mesh.vertices) == 0:
            raise ValueError("Empty mesh (no vertices)")
        if len(mesh.faces) == 0:
            raise ValueError("Empty mesh (no faces)")
        
        # Check for degenerate mesh
        if mesh.extents.max() < 1e-6:
            raise ValueError(f"Degenerate mesh with extents {mesh.extents}")
        
        # Store original bounds
        original_bounds = mesh.bounds.copy()
        original_extents = mesh.extents.copy()
        original_center = mesh.centroid.copy()
        
        # Auto-orient if enabled
        rotation_matrix = np.eye(3)
        z_adjustment = 0.0
        orientation_confidence = 0.0
        orientation_method = "none"
        
        if auto_orient:
            mesh, orientation_confidence, rotation_matrix, z_adjustment = auto_orient_mesh(mesh, logger)
            orientation_method = "stability"
        
        # Center in XY, bottom-align in Z
        mesh.vertices[:, :2] -= mesh.vertices[:, :2].mean(axis=0)
        min_z = mesh.vertices[:, 2].min()
        mesh.vertices[:, 2] -= min_z
        z_adjustment += min_z
        
        # Scale to fit in unit cube
        max_extent = mesh.extents.max()
        scale_factor = 0.95 / max_extent if max_extent > 0 else 1.0
        mesh.vertices *= scale_factor
        
        # Final validation before voxelization
        if mesh.extents.max() < 1e-6:
            raise ValueError("Mesh became degenerate after scaling")
        
        # Voxelize
        voxel_tensor = voxelize_mesh(mesh, voxel_resolution)
        
        # Validate
        validation_warnings = validate_voxel_tensor(voxel_tensor, logger)
        
        # Save voxel tensor
        np.save(output_path, voxel_tensor)
        
        # Save metadata
        metadata = {
            'source_file': str(glb_path),
            'voxel_resolution': voxel_resolution,
            'shape': list(voxel_tensor.shape),
            'occupied_voxels': int(np.sum(voxel_tensor)),
            'fill_ratio': float(np.sum(voxel_tensor) / voxel_tensor.size),
            'auto_oriented': auto_orient,
            'orientation_method': orientation_method,
            'orientation_confidence': float(orientation_confidence),
            'rotation_matrix': rotation_matrix.tolist(),
            'z_adjustment': float(z_adjustment),
            'scale_factor': float(scale_factor),
            'original_bounds': original_bounds.tolist(),
            'original_extents': original_extents.tolist(),
            'original_center': original_center.tolist(),
            'validation_warnings': validation_warnings,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return voxel_tensor, orientation_confidence, validation_warnings
        
    except Exception as e:
        raise Exception(f"Failed to process {glb_path}: {str(e)}")
    finally:
        if mesh is not None:
            del mesh
        gc.collect()


def load_file_list(file_list_path, logger):
    """
    Load the pre-generated file list from JSON.
    
    Args:
        file_list_path: Path to the JSON file containing file list
        logger: Logger object
    
    Returns:
        List of tuples: [(uid, Path), ...]
    """
    logger.info(f"Loading file list from: {file_list_path}")
    
    print(f"Loading file list from: {file_list_path}", flush=True)
    with open(file_list_path, 'r') as f:
        data = json.load(f)
    print(f"File list loaded. Total files: {len(data['files'])}", flush=True)
    
    logger.info(f"File list generated at: {data['scan_timestamp']}")
    logger.info(f"Scanned directories: {data['scan_directories']}")
    logger.info(f"Total files in list: {data['total_files']}")
    
    # Convert string paths back to Path objects
    items = [(uid, Path(path)) for uid, path in data['files']]
    
    return items


def process_single_file(args):
    """
    Process a single file in isolation to prevent memory leaks.
    This function runs in a separate process.
    """
    uid, glb_path, output_dir, check_dir, voxel_resolution, auto_orient = args
    
    # Import here to avoid sharing memory between processes
    import logging
    import sys
    
    # Set up minimal logging for this process
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.WARNING)  # Only show warnings/errors to reduce output
    
    try:
        # Create output paths
        voxel_subdir = Path(output_dir) / str(voxel_resolution)
        voxel_subdir.mkdir(parents=True, exist_ok=True)
        
        metadata_subdir = voxel_subdir / 'metadata'
        metadata_subdir.mkdir(parents=True, exist_ok=True)
        
        output_filename = uid + '.npy'
        metadata_filename = uid + '.json'
        
        output_path = voxel_subdir / output_filename
        metadata_path = metadata_subdir / metadata_filename
        
        # Check if already processed
        if check_dir:
            check_path = Path(check_dir) / str(voxel_resolution) / output_filename
        else:
            check_path = output_path
        
        if check_path.exists():
            return 'skipped', uid, None
        
        # Check if input exists
        if not glb_path.exists():
            return 'failed', uid, f"Input file not found: {glb_path}"
        
        # Convert GLB to voxel tensor with orientation
        binary_tensor, orientation_confidence, validation_warnings = glb_to_voxel_tensor(
            glb_path, output_path, metadata_path, voxel_resolution=voxel_resolution,
            auto_orient=auto_orient, logger=logger
        )
        
        # Quick validation
        occupied_voxels = np.sum(binary_tensor)
        if occupied_voxels == 0:
            return 'failed', uid, "Empty voxel tensor"
        
        # Return success with validation info
        return 'success', uid, {'warnings': len(validation_warnings), 'occupied': int(occupied_voxels)}
        
    except Exception as e:
        return 'failed', uid, str(e)
    finally:
        # Aggressive cleanup
        import gc
        gc.collect()


def process_chunk(file_list_path, output_dir, check_dir, chunk_id, total_chunks, 
                  auto_orient, voxel_resolution, logger):
    """
    Process a chunk of GLB files using pre-generated file list with multiprocessing.
    
    Uses process isolation to prevent trimesh memory leaks.
    """
    # Validate resolution
    if isinstance(voxel_resolution, int):
        voxel_resolution = (voxel_resolution,) * 3
    elif len(voxel_resolution) == 1:
        voxel_resolution = voxel_resolution * 3
    
    # Load file list from preprocessed JSON
    items = load_file_list(file_list_path, logger)
    
    total_items = len(items)
    logger.info(f"Total items in file list: {total_items}")
    
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
    del items  # Free the full list after extracting the chunk
    gc.collect()
    
    logger.info(f"Worker {chunk_id}: Processing items {start_idx} to {end_idx-1} ({len(chunk_items)} items)")
    logger.info(f"Auto-orientation: {'Enabled (stability method)' if auto_orient else 'Disabled'}")
    
    # Prepare arguments for multiprocessing
    args_list = [
        (uid, glb_path, output_dir, check_dir, voxel_resolution[0], auto_orient)
        for uid, glb_path in chunk_items
    ]
    
    del chunk_items
    gc.collect()
    
    # Statistics
    processed = 0
    skipped = 0
    failed = 0
    validation_issues_count = 0
    start_time = time.time()
    
    # Process with isolated workers (restart pool every batch to prevent memory leaks)
    batch_size = 50  # Process 50 files before restarting the pool
    total_batches = (len(args_list) + batch_size - 1) // batch_size
    
    logger.info(f"Processing {len(args_list)} files in {total_batches} batches of up to {batch_size} files each")
    
    for batch_idx in range(total_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(args_list))
        batch = args_list[batch_start:batch_end]
        
        logger.info(f"Starting batch {batch_idx + 1}/{total_batches} ({len(batch)} files)...")
        
        # Create a fresh pool for each batch to prevent memory leaks
        with mp.Pool(processes=1) as pool:
            results = pool.map(process_single_file, batch)
        
        # Process results
        for status, uid, info in results:
            if status == 'success':
                processed += 1
                if info and info.get('warnings', 0) > 0:
                    validation_issues_count += 1
            elif status == 'skipped':
                skipped += 1
            elif status == 'failed':
                failed += 1
                logger.error(f"Failed to process {uid}: {info}")
        
        # Log batch progress
        elapsed = time.time() - start_time
        items_done = batch_end
        rate = items_done / elapsed if elapsed > 0 else 0
        eta = (len(args_list) - items_done) / rate if rate > 0 else 0
        
        logger.info(f"Batch {batch_idx + 1}/{total_batches} complete. "
                   f"Progress: {items_done}/{len(args_list)} "
                   f"(Processed: {processed}, Skipped: {skipped}, Failed: {failed}) "
                   f"Rate: {rate:.2f} items/s, ETA: {eta/60:.1f} min")
        
        # Force garbage collection between batches
        gc.collect()
    
    # Final statistics
    elapsed_time = time.time() - start_time
    logger.info(f"Worker {chunk_id} completed in {elapsed_time/60:.2f} minutes")
    logger.info(f"Processed: {processed}, Skipped: {skipped}, Failed: {failed}")
    logger.info(f"Objects with validation warnings: {validation_issues_count}/{processed}")
    logger.info(f"Average rate: {(processed + skipped)/elapsed_time:.2f} items/second")
    
    return processed, skipped, failed


def main():
    parser = argparse.ArgumentParser(
        description='Voxelization worker - uses pre-generated file list with process isolation'
    )
    
    parser.add_argument('--file_list', required=True, 
                        help='Path to pre-generated file list JSON')
    parser.add_argument('--output_dir', required=True, 
                        help='Output directory for voxel tensors')
    parser.add_argument('--chunk_id', type=int, required=True, 
                        help='Worker chunk ID')
    parser.add_argument('--total_chunks', type=int, required=True, 
                        help='Total number of chunks')
    parser.add_argument('--check_dir', help='Directory to check for existing files', 
                        default=None)
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--voxel_resolution', type=int, default=64, 
                        help='Voxel grid resolution (default: 64)')
    parser.add_argument('--no_auto_orient', action='store_true', 
                        help='Disable automatic orientation')

    args = parser.parse_args()
    
    # Set check_dir to output_dir if not specified
    if args.check_dir is None:
        args.check_dir = args.output_dir
    
    # Set up logging
    logger = setup_logging()
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    logger.info("="*60)
    logger.info(f"Voxelization Worker {args.chunk_id}/{args.total_chunks}")
    logger.info(f"Using pre-generated file list: {args.file_list}")
    logger.info(f"Resolution: {args.voxel_resolution}^3")
    logger.info(f"Auto-orientation: {'Disabled' if args.no_auto_orient else 'Enabled'}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Check directory: {args.check_dir}")
    logger.info(f"Process isolation: Enabled (batch size: 50)")
    logger.info(f"Started at: {datetime.now()}")
    logger.info("="*60)
    
    try:
        # Process the chunk
        processed, skipped, failed = process_chunk(
            args.file_list,
            args.output_dir,
            args.check_dir,
            args.chunk_id,
            args.total_chunks,
            not args.no_auto_orient,
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
    
    logger.info(f"Worker {args.chunk_id} completed successfully.")
    sys.exit(0)


if __name__ == '__main__':
    main()