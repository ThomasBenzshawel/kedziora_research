#!/usr/bin/env python3
"""
Parallel Voxelization Worker
Processes a chunk of GLB files from a JSON list and converts them to voxel tensors
"""

import json
import argparse
import logging
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import trimesh
from datetime import datetime


# Fixed voxel resolution - change this as needed
VOXEL_RESOLUTION = (64, 64, 64)


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


def glb_to_voxel_tensor(glb_path, output_path, voxel_resolution=VOXEL_RESOLUTION):
    """
    Convert a GLB file to a binary voxel tensor.
    Modified from the original to be more robust for batch processing.
    """
    
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
    
    # Center the mesh at origin
    mesh.vertices -= mesh.centroid
    
    # Calculate scale factor to fit the mesh in the voxel grid
    mesh_extents = mesh.extents
    if np.any(mesh_extents == 0):
        raise ValueError("Mesh has zero extent in one or more dimensions")
    
    grid_size = min(voxel_resolution) * 0.95
    scale_factor = grid_size / max(mesh_extents)
    
    # Scale the mesh
    mesh.vertices *= scale_factor
    
    # Calculate voxel pitch
    pitch = max(mesh.extents) * 1.1 / min(voxel_resolution)
    
    # Perform voxelization
    voxel_grid = mesh.voxelized(pitch=pitch)
    
    # Fill the interior
    voxel_grid = voxel_grid.fill()
    
    # Convert to binary numpy array
    voxel_matrix = voxel_grid.matrix
    
    # Resize to match exact resolution
    if voxel_matrix.shape != voxel_resolution:
        binary_tensor = np.zeros(voxel_resolution, dtype=np.uint8)
        
        for dim in range(3):
            src_size = voxel_matrix.shape[dim]
            dst_size = voxel_resolution[dim]
            
            if src_size <= dst_size:
                dst_start = (dst_size - src_size) // 2
                dst_end = dst_start + src_size
                src_start = 0
                src_end = src_size
            else:
                src_start = (src_size - dst_size) // 2
                src_end = src_start + dst_size
                dst_start = 0
                dst_end = dst_size
            
            if dim == 0:
                src_slice_x = slice(src_start, src_end)
                dst_slice_x = slice(dst_start, dst_end)
            elif dim == 1:
                src_slice_y = slice(src_start, src_end)
                dst_slice_y = slice(dst_start, dst_end)
            else:
                src_slice_z = slice(src_start, src_end)
                dst_slice_z = slice(dst_start, dst_end)
        
        binary_tensor[dst_slice_x, dst_slice_y, dst_slice_z] = \
            voxel_matrix[src_slice_x, src_slice_y, src_slice_z].astype(np.uint8)
    else:
        binary_tensor = voxel_matrix.astype(np.uint8)
    
    # Save to file
    np.save(output_path, binary_tensor)
    
    return binary_tensor


def process_chunk(json_path, scan_dir, output_dir, check_dir, chunk_id, total_chunks, logger):
    """
    Process a chunk of the GLB files
    """
    
    # Load the JSON file
    logger.info(f"Loading JSON from {json_path}")
    with open(json_path, 'r') as f:
        file_paths = json.load(f)
    
    # Convert to list of (uid, path) tuples for consistent ordering
    if isinstance(file_paths, dict):
        items = list(file_paths.items())
    else:
        # Assume it's already a list of paths
        items = [(str(i), path) for i, path in enumerate(file_paths)]
    
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
    
    # Statistics
    processed = 0
    skipped = 0
    failed = 0
    start_time = time.time()
    
    for idx, (uid, relative_path) in enumerate(chunk_items):
        try:
            # Construct paths
            glb_path = Path(scan_dir) / relative_path
            
            # Maintain directory structure for output
            # Extract the directory structure from the relative path
            rel_path = Path(relative_path)
            output_subdir = Path(output_dir) / rel_path.parent
            output_subdir.mkdir(parents=True, exist_ok=True)
            
            # Output file path (change extension from .glb to .npy)
            output_filename = rel_path.stem + '.npy'
            output_path = output_subdir / output_filename
            
            # Check if already processed
            if check_dir:
                check_path = Path(check_dir) / rel_path.parent / output_filename
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
            
            # Convert GLB to voxel tensor
            binary_tensor = glb_to_voxel_tensor(glb_path, output_path)
            
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
    
    # Final statistics
    elapsed_time = time.time() - start_time
    logger.info(f"Worker {chunk_id} completed in {elapsed_time/60:.2f} minutes")
    logger.info(f"Processed: {processed}, Skipped: {skipped}, Failed: {failed}")
    logger.info(f"Average rate: {(processed + skipped)/elapsed_time:.2f} items/second")
    
    return processed, skipped, failed


def main():
    parser = argparse.ArgumentParser(description='Voxelization worker for parallel processing')
    
    parser.add_argument('--json_path', required=True, help='Path to JSON file with GLB paths')
    parser.add_argument('--scan_dir', required=True, help='Base directory for GLB files')
    parser.add_argument('--output_dir', required=True, help='Output directory for voxel tensors')
    parser.add_argument('--chunk_id', type=int, required=True, help='Worker chunk ID')
    parser.add_argument('--total_chunks', type=int, required=True, help='Total number of chunks')
    parser.add_argument('--check_dir', help='Directory to check for existing files (default: output_dir)')
    parser.add_argument('--log_file', help='Log file path')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging(args.log_file)
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    logger.info("="*60)
    logger.info(f"Voxelization Worker {args.chunk_id}/{args.total_chunks}")
    logger.info(f"Resolution: {VOXEL_RESOLUTION}")
    logger.info(f"Started at: {datetime.now()}")
    logger.info("="*60)
    
    try:
        # Process the chunk
        processed, skipped, failed = process_chunk(
            args.json_path,
            args.scan_dir,
            args.output_dir,
            args.check_dir,
            args.chunk_id,
            args.total_chunks,
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