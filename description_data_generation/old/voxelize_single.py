#!/usr/bin/env python3
"""
GLB to Voxel Tensor Converter with Visualization
Converts .glb 3D model files to binary voxel tensors saved as .npy files
and provides visualization capabilities to verify results
"""

import numpy as np
import trimesh
import argparse
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def glb_to_voxel_tensor(glb_path, voxel_resolution=(64, 64, 64), output_path=None):
    """
    Convert a GLB file to a binary voxel tensor.
    
    Args:
        glb_path (str or Path): Path to the input .glb file
        voxel_resolution (tuple): Dimensions of the voxel grid (x, y, z)
        output_path (str or Path, optional): Path for output .npy file
                                            If None, uses input filename with .npy extension
    
    Returns:
        numpy.ndarray: Binary voxel tensor with shape equal to voxel_resolution
    """
    
    # Convert paths to Path objects
    glb_path = Path(glb_path)
    
    if not glb_path.exists():
        raise FileNotFoundError(f"GLB file not found: {glb_path}")
    
    if not glb_path.suffix.lower() == '.glb':
        print(f"Warning: File extension is not .glb, attempting to load anyway...")
    
    # Load the GLB file
    print(f"Loading GLB file: {glb_path}")
    scene = trimesh.load(glb_path, force='scene')
    
    # Combine all meshes in the scene into a single mesh
    if isinstance(scene, trimesh.Scene):
        # Get all geometry from the scene
        meshes = []
        for name, geometry in scene.geometry.items():
            if isinstance(geometry, trimesh.Trimesh):
                meshes.append(geometry)
        
        if len(meshes) == 0:
            raise ValueError("No valid mesh geometry found in the GLB file")
        
        # Combine all meshes
        if len(meshes) == 1:
            mesh = meshes[0]
        else:
            print(f"Combining {len(meshes)} mesh objects...")
            mesh = trimesh.util.concatenate(meshes)
    elif isinstance(scene, trimesh.Trimesh):
        mesh = scene
    else:
        raise ValueError(f"Unexpected type loaded from GLB: {type(scene)}")
    
    print(f"Mesh statistics:")
    print(f"  - Vertices: {len(mesh.vertices)}")
    print(f"  - Faces: {len(mesh.faces)}")
    print(f"  - Bounding box: {mesh.bounds[0]} to {mesh.bounds[1]}")
    
    # Center the mesh at origin
    mesh.vertices -= mesh.centroid
    
    # Calculate scale factor to fit the mesh in the voxel grid while maintaining proportions
    # Leave a small margin (95% of grid size) to ensure no clipping
    mesh_extents = mesh.extents
    grid_size = min(voxel_resolution) * 0.95
    scale_factor = grid_size / max(mesh_extents)
    
    # Scale the mesh
    mesh.vertices *= scale_factor
    
    print(f"Preprocessing:")
    print(f"  - Centered at origin")
    print(f"  - Scaled by factor: {scale_factor:.4f}")
    print(f"  - New bounding box: {mesh.bounds[0]} to {mesh.bounds[1]}")
    
    # Create voxel grid
    print(f"Voxelizing to {voxel_resolution[0]}x{voxel_resolution[1]}x{voxel_resolution[2]} grid...")
    
    # Calculate voxel pitch (size of each voxel)
    # The grid should span slightly larger than the mesh to ensure everything fits
    pitch = max(mesh.extents) * 1.1 / min(voxel_resolution)
    
    # Perform voxelization
    voxel_grid = mesh.voxelized(pitch=pitch)
    
    # Fill the interior (solid voxelization)
    voxel_grid = voxel_grid.fill()
    
    # Convert to binary numpy array
    # The matrix from trimesh needs to be converted to our desired resolution
    voxel_matrix = voxel_grid.matrix
    
    # Resize if necessary to match exact resolution
    if voxel_matrix.shape != voxel_resolution:
        # Pad or crop to match the desired resolution
        binary_tensor = np.zeros(voxel_resolution, dtype=np.uint8)
        
        # Calculate the region to copy (centered)
        for dim in range(3):
            src_size = voxel_matrix.shape[dim]
            dst_size = voxel_resolution[dim]
            
            if src_size <= dst_size:
                # Pad: center the source in destination
                dst_start = (dst_size - src_size) // 2
                dst_end = dst_start + src_size
                src_start = 0
                src_end = src_size
            else:
                # Crop: take center of source
                src_start = (src_size - dst_size) // 2
                src_end = src_start + dst_size
                dst_start = 0
                dst_end = dst_size
            
            # Create slice objects for each dimension
            if dim == 0:
                src_slice_x = slice(src_start, src_end)
                dst_slice_x = slice(dst_start, dst_end)
            elif dim == 1:
                src_slice_y = slice(src_start, src_end)
                dst_slice_y = slice(dst_start, dst_end)
            else:
                src_slice_z = slice(src_start, src_end)
                dst_slice_z = slice(dst_start, dst_end)
        
        # Copy the data
        binary_tensor[dst_slice_x, dst_slice_y, dst_slice_z] = \
            voxel_matrix[src_slice_x, src_slice_y, src_slice_z].astype(np.uint8)
    else:
        binary_tensor = voxel_matrix.astype(np.uint8)
    
    # Save to file
    if output_path is None:
        output_path = glb_path.with_suffix('.npy')
    else:
        output_path = Path(output_path)
    
    np.save(output_path, binary_tensor)
    print(f"Saved voxel tensor to: {output_path}")
    
    # Print statistics
    occupied_voxels = np.sum(binary_tensor)
    total_voxels = np.prod(voxel_resolution)
    occupancy_rate = occupied_voxels / total_voxels * 100
    
    print(f"Voxel tensor statistics:")
    print(f"  - Shape: {binary_tensor.shape}")
    print(f"  - Occupied voxels: {occupied_voxels:,} / {total_voxels:,} ({occupancy_rate:.2f}%)")
    print(f"  - Data type: {binary_tensor.dtype}")
    print(f"  - Memory size: {binary_tensor.nbytes / 1024:.2f} KB")
    
    return binary_tensor


def visualize_voxel_tensor(voxel_tensor, output_path=None, show_plot=True, max_voxels=50000):
    """
    Visualize a voxel tensor as a 3D plot.
    
    Args:
        voxel_tensor (numpy.ndarray): Binary voxel tensor to visualize
        output_path (str or Path, optional): Path to save the visualization image
        show_plot (bool): Whether to display the plot interactively
        max_voxels (int): Maximum number of voxels to render (for performance)
    """
    
    print(f"Visualizing voxel tensor with shape: {voxel_tensor.shape}")
    
    # Find occupied voxels
    occupied_coords = np.where(voxel_tensor > 0)
    num_occupied = len(occupied_coords[0])
    
    print(f"Found {num_occupied:,} occupied voxels")
    
    if num_occupied == 0:
        print("Warning: No occupied voxels found in tensor")
        return
    
    # Subsample if too many voxels for performance
    if num_occupied > max_voxels:
        print(f"Subsampling to {max_voxels} voxels for performance...")
        indices = np.random.choice(num_occupied, max_voxels, replace=False)
        x_coords = occupied_coords[0][indices]
        y_coords = occupied_coords[1][indices]
        z_coords = occupied_coords[2][indices]
    else:
        x_coords = occupied_coords[0]
        y_coords = occupied_coords[1]
        z_coords = occupied_coords[2]
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot voxels as scatter points
    scatter = ax.scatter(x_coords, y_coords, z_coords, 
                        c=z_coords, cmap='viridis', 
                        alpha=0.6, s=20)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Voxelized 3D Model\n{voxel_tensor.shape} grid, {len(x_coords):,} voxels shown')
    
    # Add colorbar
    plt.colorbar(scatter, ax=ax, shrink=0.8, label='Z-coordinate')
    
    # Set equal aspect ratio
    max_range = max(voxel_tensor.shape)
    ax.set_xlim(0, max_range)
    ax.set_ylim(0, max_range)
    ax.set_zlim(0, max_range)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Save if requested
    if output_path:
        output_path = Path(output_path)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {output_path}")
    
    # Show plot
    if show_plot:
        plt.show()
    else:
        plt.close()


def load_and_visualize_voxels(npy_path, output_image=None, show_plot=True):
    """
    Load a voxel tensor from .npy file and visualize it.
    
    Args:
        npy_path (str or Path): Path to the .npy file containing voxel tensor
        output_image (str or Path, optional): Path to save visualization image
        show_plot (bool): Whether to display the plot interactively
    
    Returns:
        numpy.ndarray: The loaded voxel tensor
    """
    
    npy_path = Path(npy_path)
    
    if not npy_path.exists():
        raise FileNotFoundError(f"Voxel file not found: {npy_path}")
    
    print(f"Loading voxel tensor from: {npy_path}")
    
    # Load the voxel tensor
    voxel_tensor = np.load(npy_path)
    
    print(f"Loaded tensor statistics:")
    print(f"  - Shape: {voxel_tensor.shape}")
    print(f"  - Data type: {voxel_tensor.dtype}")
    print(f"  - Occupied voxels: {np.sum(voxel_tensor):,}")
    print(f"  - File size: {npy_path.stat().st_size / 1024:.2f} KB")
    
    # Visualize
    visualize_voxel_tensor(voxel_tensor, output_image, show_plot)
    
    return voxel_tensor


def parse_resolution(resolution_str):
    """Parse resolution string like '64x64x64' or '64' into tuple."""
    parts = resolution_str.split('x')
    if len(parts) == 1:
        # Single number provided, use for all dimensions
        size = int(parts[0])
        return (size, size, size)
    elif len(parts) == 3:
        # Full resolution specified
        return tuple(int(p) for p in parts)
    else:
        raise ValueError(f"Invalid resolution format: {resolution_str}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert GLB files to binary voxel tensors and visualize results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert with default 64x64x64 resolution and show visualization
  python glb_to_voxel.py model.glb --visualize
  
  # Convert with custom resolution
  python glb_to_voxel.py model.glb --resolution 128x128x128
  
  # Convert and save visualization as image
  python glb_to_voxel.py model.glb --visualize --save-viz model_voxels.png
  
  # Only visualize existing voxel file
  python glb_to_voxel.py --load-voxels model.npy --visualize
  
  # Batch convert multiple files
  python glb_to_voxel.py *.glb --resolution 64 --visualize
        """
    )
    
    parser.add_argument(
        'input_files',
        nargs='*',
        help='Input GLB file(s) to convert'
    )
    
    parser.add_argument(
        '--resolution', '-r',
        default='64',
        help='Voxel grid resolution (e.g., "64" for 64x64x64 or "32x64x128" for non-uniform)'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Output path for .npy file (only valid with single input file)'
    )
    
    parser.add_argument(
        '--visualize', '-v',
        action='store_true',
        help='Show 3D visualization of the voxelized result'
    )
    
    parser.add_argument(
        '--save-viz',
        help='Save visualization as image (PNG, JPG, etc.)'
    )
    
    parser.add_argument(
        '--load-voxels',
        help='Load and visualize existing voxel .npy file instead of converting GLB'
    )
    
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Don\'t show interactive plot (only save to file)'
    )
    
    args = parser.parse_args()
    
    # Handle load-voxels mode
    if args.load_voxels:
        if args.input_files:
            print("Warning: input files ignored when using --load-voxels", file=sys.stderr)
        
        try:
            load_and_visualize_voxels(
                args.load_voxels, 
                args.save_viz, 
                not args.no_display
            )
        except Exception as e:
            print(f"Error loading voxels: {e}", file=sys.stderr)
            sys.exit(1)
        
        return
    
    # Normal GLB conversion mode
    if not args.input_files:
        print("Error: No input files specified", file=sys.stderr)
        parser.print_help()
        sys.exit(1)
    
    # Parse resolution
    try:
        resolution = parse_resolution(args.resolution)
        print(f"Using voxel resolution: {resolution[0]}x{resolution[1]}x{resolution[2]}")
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Check output path compatibility
    if args.output and len(args.input_files) > 1:
        print("Error: --output can only be used with a single input file", file=sys.stderr)
        sys.exit(1)
    
    # Process each file
    failed_files = []
    for input_file in args.input_files:
        print(f"\n{'='*60}")
        print(f"Processing: {input_file}")
        print('='*60)
        
        try:
            # Convert to voxels
            voxel_tensor = glb_to_voxel_tensor(
                input_file,
                voxel_resolution=resolution,
                output_path=args.output if len(args.input_files) == 1 else None
            )
            
            # Visualize if requested
            if args.visualize:
                print(f"\nVisualizing voxelized result...")
                
                # Determine output image path for visualization
                viz_output = None
                if args.save_viz:
                    if len(args.input_files) == 1:
                        viz_output = args.save_viz
                    else:
                        # Multiple files - generate unique names
                        input_path = Path(input_file)
                        viz_ext = Path(args.save_viz).suffix or '.png'
                        viz_output = input_path.with_suffix(f'_voxels{viz_ext}')
                
                visualize_voxel_tensor(
                    voxel_tensor, 
                    viz_output, 
                    not args.no_display
                )
                
        except Exception as e:
            print(f"Error processing {input_file}: {e}", file=sys.stderr)
            failed_files.append(input_file)
    
    # Summary
    if len(args.input_files) > 1:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print('='*60)
        print(f"Processed: {len(args.input_files) - len(failed_files)}/{len(args.input_files)} files")
        if failed_files:
            print("Failed files:")
            for f in failed_files:
                print(f"  - {f}")
    
    sys.exit(len(failed_files))


if __name__ == "__main__":
    main()