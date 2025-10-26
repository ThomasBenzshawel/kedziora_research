#!/usr/bin/env python3
"""
Batch Voxel Visualizer
Loads voxel tensors from .npy files and generates visualization images
"""

import numpy as np
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for batch processing
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
from tqdm import tqdm


def visualize_voxel_tensor(voxel_tensor, output_path, view_angle=(30, 45), 
                          max_voxels=50000, figsize=(10, 10), dpi=150,
                          colormap='viridis', background_color='white',
                          point_size=20, alpha=0.6):
    """
    Create and save a visualization of the voxel tensor.
    
    Args:
        voxel_tensor: Binary voxel tensor
        output_path: Path to save the image
        view_angle: Tuple of (elevation, azimuth) angles for camera
        max_voxels: Maximum voxels to render for performance
        figsize: Figure size in inches
        dpi: Resolution of saved image
        colormap: Matplotlib colormap name
        background_color: Background color
        point_size: Size of voxel points in scatter plot
        alpha: Transparency of voxels
    """
    output_path = Path(output_path)
    
    # Find occupied voxels
    occupied_coords = np.where(voxel_tensor > 0)
    num_occupied = len(occupied_coords[0])
    
    if num_occupied == 0:
        # Create empty plot with message
        fig = plt.figure(figsize=figsize, facecolor=background_color)
        ax = fig.add_subplot(111, projection='3d')
        ax.text2D(0.5, 0.5, "No voxels found", transform=ax.transAxes,
                  ha='center', va='center', fontsize=20)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor=background_color)
        plt.close()
        return {'status': 'empty', 'voxel_count': 0}
    
    # Subsample if too many voxels for performance
    if num_occupied > max_voxels:
        indices = np.random.choice(num_occupied, max_voxels, replace=False)
        x_coords = occupied_coords[0][indices]
        y_coords = occupied_coords[1][indices]
        z_coords = occupied_coords[2][indices]
        rendered_voxels = max_voxels
    else:
        x_coords = occupied_coords[0]
        y_coords = occupied_coords[1]
        z_coords = occupied_coords[2]
        rendered_voxels = num_occupied
    
    # Create figure
    fig = plt.figure(figsize=figsize, facecolor=background_color)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot voxels as scatter points
    scatter = ax.scatter(x_coords, y_coords, z_coords, 
                        c=z_coords, cmap=colormap, 
                        alpha=alpha, s=point_size, edgecolors='none')
    
    # Set viewing angle
    ax.view_init(elev=view_angle[0], azim=view_angle[1])
    
    # Set labels
    ax.set_xlabel('X', fontsize=10)
    ax.set_ylabel('Y', fontsize=10)
    ax.set_zlabel('Z', fontsize=10)
    
    # Set equal aspect ratio and limits
    max_range = max(voxel_tensor.shape)
    ax.set_xlim(0, max_range)
    ax.set_ylim(0, max_range)
    ax.set_zlim(0, max_range)
    
    # Styling options
    ax.grid(True, alpha=0.3)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    # Save the figure
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', 
                facecolor=background_color, edgecolor='none')
    plt.close()
    
    return {
        'status': 'success',
        'voxel_count': num_occupied,
        'rendered_voxels': rendered_voxels
    }


def process_voxel_folder(input_folder, output_folder=None, image_format='png',
                        view_angles=None, max_voxels=50000, dpi=150,
                        colormap='viridis', figsize=(10, 10), point_size=20,
                        alpha=0.6, background_color='white'):
    """
    Process all .npy voxel files in a folder and generate visualizations.
    
    Args:
        input_folder: Path to folder containing .npy voxel files
        output_folder: Output folder for images
        image_format: Image format (png, jpg, etc.)
        view_angles: List of (elevation, azimuth) tuples for multiple views
        max_voxels: Maximum voxels to render per image
        dpi: Image resolution
        colormap: Matplotlib colormap
        figsize: Figure size as tuple
        point_size: Size of voxel points
        alpha: Transparency
        background_color: Background color
    
    Returns:
        Dictionary with processing results
    """
    input_folder = Path(input_folder)
    
    if not input_folder.exists():
        raise ValueError(f"Input folder not found: {input_folder}")
    
    # Set default output folder
    if output_folder is None:
        output_folder = input_folder / 'voxel_visualizations'
    else:
        output_folder = Path(output_folder)
    
    # Create output directory
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Default view angles if not specified
    if view_angles is None:
        view_angles = [(30, 45)]  # Single default view
    
    # Find all .npy files
    npy_files = list(input_folder.glob('*.npy'))
    
    if not npy_files:
        print(f"No .npy files found in {input_folder}")
        return {}
    
    print(f"Found {len(npy_files)} voxel files to visualize")
    print(f"Output folder: {output_folder}")
    print(f"Generating {len(view_angles)} view(s) per voxel file")
    print(f"Image settings: {dpi} DPI, {figsize} figure size")
    
    # Process results
    results = {
        'processed': [],
        'failed': [],
        'settings': {
            'views': view_angles,
            'max_voxels': max_voxels,
            'colormap': colormap,
            'dpi': dpi,
            'figsize': figsize,
            'point_size': point_size,
            'alpha': alpha
        }
    }
    
    # Process each file with progress bar
    for npy_file in tqdm(npy_files, desc="Generating visualizations"):
        try:
            # Load voxel tensor
            voxel_tensor = np.load(npy_file)
            
            # Get file info
            file_info = {
                'name': npy_file.name,
                'shape': voxel_tensor.shape,
                'dtype': str(voxel_tensor.dtype),
                'visualizations': []
            }
            
            # Calculate statistics
            if voxel_tensor.dtype == bool or np.issubdtype(voxel_tensor.dtype, np.integer):
                occupied_voxels = np.sum(voxel_tensor > 0)
                total_voxels = np.prod(voxel_tensor.shape)
                occupancy_rate = (occupied_voxels / total_voxels * 100) if total_voxels > 0 else 0
                
                file_info['occupied_voxels'] = int(occupied_voxels)
                file_info['total_voxels'] = int(total_voxels)
                file_info['occupancy_rate'] = round(occupancy_rate, 2)
            
            # Generate visualizations for each view angle
            for i, (elev, azim) in enumerate(view_angles):
                if len(view_angles) == 1:
                    image_name = f"{npy_file.stem}.{image_format}"
                else:
                    image_name = f"{npy_file.stem}_view{i+1}_e{elev}_a{azim}.{image_format}"
                
                image_path = output_folder / image_name
                
                viz_result = visualize_voxel_tensor(
                    voxel_tensor, 
                    image_path,
                    view_angle=(elev, azim),
                    max_voxels=max_voxels,
                    figsize=figsize,
                    dpi=dpi,
                    colormap=colormap,
                    background_color=background_color,
                    point_size=point_size,
                    alpha=alpha
                )
                
                file_info['visualizations'].append({
                    'file': image_name,
                    'view': {'elevation': elev, 'azimuth': azim},
                    'status': viz_result['status'],
                    'rendered_voxels': viz_result.get('rendered_voxels', 0)
                })
            
            results['processed'].append(file_info)
            
        except Exception as e:
            results['failed'].append({
                'name': npy_file.name,
                'error': str(e)
            })
            print(f"\nError processing {npy_file.name}: {e}")
    
    # Save processing report
    report_path = output_folder / 'visualization_report.json'
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE")
    print("="*60)
    print(f"Successfully processed: {len(results['processed'])}/{len(npy_files)} files")
    print(f"Failed: {len(results['failed'])} files")
    print(f"Images saved to: {output_folder}")
    print(f"Report saved to: {report_path}")
    
    if results['failed']:
        print("\nFailed files:")
        for failed in results['failed']:
            print(f"  - {failed['name']}: {failed['error']}")
    
    # Print statistics
    if results['processed']:
        total_voxels_processed = sum(
            f.get('occupied_voxels', 0) for f in results['processed']
        )
        print(f"\nTotal occupied voxels visualized: {total_voxels_processed:,}")
    
    return results


def parse_view_angles(view_strings):
    """Parse view angle strings into (elevation, azimuth) tuples."""
    views = []
    for view_str in view_strings:
        parts = view_str.split(',')
        if len(parts) != 2:
            raise ValueError(f"Invalid view format: {view_str}. Expected 'elevation,azimuth'")
        try:
            elev = float(parts[0])
            azim = float(parts[1])
            views.append((elev, azim))
        except ValueError:
            raise ValueError(f"Invalid view angles: {view_str}. Must be numeric.")
    return views


def main():
    parser = argparse.ArgumentParser(
        description='Generate visualizations for voxel tensors stored in .npy files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize all .npy files in current directory
  python voxel_visualizer.py .
  
  # Custom output folder and high quality images
  python voxel_visualizer.py /path/to/voxels -o ./images --dpi 300
  
  # Generate multiple views per voxel file
  python voxel_visualizer.py ./voxels --views 30,45 20,90 45,0 60,120
  
  # Custom appearance settings
  python voxel_visualizer.py ./voxels --colormap plasma --size 30 --alpha 0.8
  
  # Large figure size for presentations
  python voxel_visualizer.py ./voxels --figsize 15 15 --dpi 200
        """
    )
    
    parser.add_argument(
        'input_folder',
        help='Folder containing .npy voxel files'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='Output folder for visualization images (default: input_folder/voxel_visualizations)'
    )
    
    parser.add_argument(
        '--views',
        nargs='+',
        help='View angles as "elevation,azimuth" pairs (default: "30,45")'
    )
    
    parser.add_argument(
        '--format',
        default='png',
        choices=['png', 'jpg', 'jpeg', 'svg', 'pdf'],
        help='Image format (default: png)'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='Image resolution in DPI (default: 150)'
    )
    
    parser.add_argument(
        '--figsize',
        nargs=2,
        type=float,
        default=[10, 10],
        help='Figure size in inches, width height (default: 10 10)'
    )
    
    parser.add_argument(
        '--max-voxels',
        type=int,
        default=50000,
        help='Maximum voxels to render per image (default: 50000)'
    )
    
    parser.add_argument(
        '--colormap',
        default='viridis',
        help='Matplotlib colormap name (default: viridis). Try: plasma, coolwarm, turbo, rainbow'
    )
    
    parser.add_argument(
        '--size',
        type=float,
        default=20,
        help='Size of voxel points (default: 20)'
    )
    
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.6,
        help='Transparency of voxels, 0=transparent, 1=opaque (default: 0.6)'
    )
    
    parser.add_argument(
        '--background',
        default='white',
        help='Background color (default: white). Can use color names or hex codes'
    )
    
    args = parser.parse_args()
    
    # Parse view angles if provided
    view_angles = None
    if args.views:
        try:
            view_angles = parse_view_angles(args.views)
        except ValueError as e:
            print(f"Error: {e}")
            return 1
    
    # Run the batch processing
    try:
        results = process_voxel_folder(
            input_folder=args.input_folder,
            output_folder=args.output,
            image_format=args.format,
            view_angles=view_angles,
            max_voxels=args.max_voxels,
            dpi=args.dpi,
            colormap=args.colormap,
            figsize=tuple(args.figsize),
            point_size=args.size,
            alpha=args.alpha,
            background_color=args.background
        )
        
        # Return non-zero exit code if any files failed
        return len(results.get('failed', []))
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())