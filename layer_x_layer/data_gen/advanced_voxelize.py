import os
import point_cloud_utils as pcu
import numpy as np
from tqdm import tqdm
import fvdb
import torch
import argparse
import psutil
import gc
import math
from concurrent.futures import ProcessPoolExecutor
import time
import queue
import threading

# Add memory monitoring functions
def log_memory_usage(checkpoint_name):
    """Log both RAM and GPU memory usage at checkpoints"""
    # RAM usage
    process = psutil.Process(os.getpid())
    ram_usage = process.memory_info().rss / (1024 * 1024 * 1024)  # GB
    
    # GPU usage if available
    gpu_allocated = 0
    gpu_reserved = 0
    if torch.cuda.is_available():
        gpu_allocated = torch.cuda.memory_allocated() / (1024 * 1024 * 1024)  # GB
        gpu_reserved = torch.cuda.memory_reserved() / (1024 * 1024 * 1024)  # GB
    
    print(f"[MEMORY] {checkpoint_name} - RAM: {ram_usage:.2f} GB | GPU Allocated: {gpu_allocated:.2f} GB | GPU Reserved: {gpu_reserved:.2f} GB")
    return ram_usage, gpu_allocated

def clean_memory():
    """Clean up memory (both CPU and GPU)"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log_memory_usage("After cleanup")

def calculate_mesh_surface_area(v, f):
    """Calculate surface area of a mesh"""
    # Get all triangle vertices
    triangle_vertices = v[f]
    
    # Calculate edge vectors
    edge1 = triangle_vertices[:, 1] - triangle_vertices[:, 0]
    edge2 = triangle_vertices[:, 2] - triangle_vertices[:, 0]
    
    # Calculate cross product
    cross = np.cross(edge1, edge2)
    
    # Calculate area of each triangle
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    
    # Sum all areas
    return np.sum(areas)

def calculate_optimal_sample_points(v, f, max_sample_points=5_000_000, min_sample_points=100_000):
    """Calculate optimal sample count based on mesh complexity"""
    face_count = f.shape[0]
    
    try:
        surface_area = calculate_mesh_surface_area(v, f)
        
        # Base heuristic: more faces and larger surface area need more points
        # Scale sample count with face count, but with diminishing returns
        face_factor = math.log10(face_count + 1) / 5  # Normalize
        
        # Scale with surface area, with upper and lower bounds
        # Assuming "typical" mesh has surface area around 1.0 units
        density_factor = min(2.0, max(0.1, math.sqrt(surface_area) / 2))
        
        # Calculate sample points with heuristic
        sample_points = int(min(max_sample_points, 
                               max(min_sample_points, 
                                   min_sample_points * face_factor * density_factor)))
        
        print(f"Adaptive sampling: {sample_points} points (face_factor={face_factor:.2f}, density_factor={density_factor:.2f})")
        return sample_points
    
    except Exception as e:
        print(f"Error calculating optimal sample points: {e}")
        # Fallback to default
        return min(max_sample_points, max(min_sample_points, face_count * 10))

# Prefetching mechanism using a separate thread
class MeshPrefetcher:
    def __init__(self, model_ids, category_dir, prefetch_size=2):
        self.model_ids = model_ids
        self.category_dir = category_dir
        self.prefetch_size = prefetch_size
        self.queue = queue.Queue(maxsize=prefetch_size)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._prefetch_worker)
        self.thread.daemon = True
        self.thread.start()
    
    def _prefetch_worker(self):
        """Thread worker to load meshes in advance"""
        try:
            for model_id in self.model_ids:
                if self.stop_event.is_set():
                    break
                    
                try:
                    model_path = os.path.join(self.category_dir, model_id)
                    v, f = pcu.load_mesh_vf(model_path)
                    self.queue.put((model_id, v, f))
                except Exception as e:
                    print(f"Error prefetching {model_id}: {e}")
                    # Put None to indicate error
                    self.queue.put((model_id, None, None))
            
            # Signal end of data
            self.queue.put((None, None, None))
        except Exception as e:
            print(f"Prefetcher thread crashed: {e}")
            # Ensure we signal end of data
            self.queue.put((None, None, None))
    
    def get_next(self):
        """Get the next prefetched mesh with timeout"""
        try:
            return self.queue.get(timeout=60)  # 1-minute timeout
        except queue.Empty:
            print("Warning: Prefetcher queue timeout, returning None")
            return None, None, None
    
    def stop(self):
        """Stop the prefetcher thread"""
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)

# Process point clouds in chunks
def process_point_cloud_in_chunks(ref_xyz, ref_normal, function, chunk_size=500000):
    """Process large point clouds in manageable chunks"""
    results = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    total_chunks = math.ceil(ref_xyz.shape[0] / chunk_size)
    
    for i in range(0, ref_xyz.shape[0], chunk_size):
        end = min(i + chunk_size, ref_xyz.shape[0])
        
        try:
            # Create chunk tensors directly on the target device
            with torch.no_grad():
                if isinstance(ref_xyz, np.ndarray):
                    xyz_chunk = torch.tensor(ref_xyz[i:end], dtype=torch.float32, device=device)
                    normal_chunk = torch.tensor(ref_normal[i:end], dtype=torch.float32, device=device)
                else:
                    # Use clone to avoid tracking history
                    xyz_chunk = ref_xyz[i:end].clone().to(device)
                    normal_chunk = ref_normal[i:end].clone().to(device)
            
            # Process chunk
            chunk_result = function(xyz_chunk, normal_chunk)
            results.append(chunk_result)
        except Exception as e:
            print(f"Error processing chunk {i//chunk_size + 1}/{total_chunks}: {e}")
            # Continue with other chunks
        finally:
            # Clean intermediate results on GPU
            if 'xyz_chunk' in locals(): del xyz_chunk
            if 'normal_chunk' in locals(): del normal_chunk
            if device == 'cuda':
                torch.cuda.empty_cache()
    
    if not results:
        print("Warning: All chunks failed processing")
    
    return results

# Set up common buffers for reuse with preallocation
class ProcessingBuffers:
    def __init__(self, max_points=1_000_000):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.max_points = max_points
        
        # Initialize buffers with small size initially to save memory
        initial_size = min(100000, max_points)
        
        # Preallocate buffers on the correct device
        with torch.no_grad():
            self.point_buffer = torch.zeros((initial_size, 3), dtype=torch.float32, device=self.device)
            self.normal_buffer = torch.zeros((initial_size, 3), dtype=torch.float32, device=self.device)
    
    def get_buffers(self, num_points):
        """Get appropriately sized buffers for the given number of points"""
        # Resize buffers if needed
        if self.point_buffer.shape[0] < num_points:
            new_size = min(self.max_points, max(num_points, int(self.point_buffer.shape[0] * 1.5)))
            print(f"Resizing buffers from {self.point_buffer.shape[0]} to {new_size} points")
            
            # Release old buffers before allocating new ones
            del self.point_buffer, self.normal_buffer
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            
            try:
                with torch.no_grad():
                    self.point_buffer = torch.zeros((new_size, 3), dtype=torch.float32, device=self.device)
                    self.normal_buffer = torch.zeros((new_size, 3), dtype=torch.float32, device=self.device)
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    print(f"GPU OOM when allocating buffers of size {new_size}, falling back to CPU")
                    self.device = 'cpu'  # Fallback to CPU
                    with torch.no_grad():
                        self.point_buffer = torch.zeros((new_size, 3), dtype=torch.float32, device=self.device)
                        self.normal_buffer = torch.zeros((new_size, 3), dtype=torch.float32, device=self.device)
                else:
                    raise
        
        return self.point_buffer[:num_points], self.normal_buffer[:num_points]
    
    def clear(self):
        """Release buffers to free memory"""
        del self.point_buffer, self.normal_buffer
        if self.device == 'cuda':
            torch.cuda.empty_cache()


def process_single_model(model_data, target_dir, args, buffers=None):
    """Process a single model with memory optimization"""
    start_time = time.time()
    """Process a single model with memory optimization"""
    model_id, v, f = model_data
    
    if v is None or f is None:
        print(f"Skipping {model_id} due to loading error")
        return False
    
    target_path = os.path.join(target_dir, f"{model_id.split('.')[0]}.pkl")
    if os.path.exists(target_path):
        print(f"Target already exists: {target_path}, skipping")
        return False
    
    log_memory_usage(f"Before processing model {model_id}")
    
    try:
        # Determine optimal sample size using adaptive sampling
        if args.use_dynamic_sampling:
            adjusted_sample_size = calculate_optimal_sample_points(
                v, f, max_sample_points=args.max_sample_points, min_sample_points=100_000)
        else:
            if args.num_vox > 512:
                adjusted_sample_size = min(args.max_sample_points, 5_000_000)
            else:
                adjusted_sample_size = min(args.max_sample_points, 1_000_000)
        
        try:
            # Sample points from mesh
            fid, bc = pcu.sample_mesh_random(v, f, adjusted_sample_size)
            ref_xyz = pcu.interpolate_barycentric_coords(f, fid, bc, v)
        except Exception as e:
            print(f"First sampling attempt failed: {e}, retrying with reduced sampling")
            fallback_sample_size = adjusted_sample_size // 2
            print(f"Fallback to {fallback_sample_size} sample points")
            fid, bc = pcu.sample_mesh_random(v, f, fallback_sample_size)
            ref_xyz = pcu.interpolate_barycentric_coords(f, fid, bc, v)
        
        log_memory_usage(f"After sampling {model_id}")
        
        # Estimate normals
        n = pcu.estimate_mesh_face_normals(v, f)
        ref_normal = n[fid]
        log_memory_usage(f"After normal estimation {model_id}")
        
        # Set voxel size based on resolution
        num_vox = args.num_vox
        max_num_vox = max(num_vox, 512)
        vox_size = 1.0 / max_num_vox
        
        # Calculate voxelization on CPU
        ijk = pcu.voxelize_triangle_mesh(v, f.astype(np.int32), vox_size, np.zeros(3))
        log_memory_usage(f"After voxelization {model_id}")
        
        # Ensure ijk has the correct integer type
        if not np.issubdtype(ijk.dtype, np.integer):
            print(f"Warning: ijk has non-integer type {ijk.dtype}, converting to int32")
            ijk = ijk.astype(np.int32)
        
        # Convert to tensor with direct device placement to avoid redundant transfers
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        with torch.no_grad():
            ijk_tensor = torch.tensor(ijk, dtype=torch.int32, device=device)
        
        # Free CPU memory immediately after tensor creation
        del ijk
        gc.collect()
        
        log_memory_usage(f"After creating tensor {model_id}")
        
        try:
            # Create grid 
            if device == 'cuda':
                grid = fvdb.gridbatch_from_ijk(fvdb.JaggedTensor([ijk_tensor]), 
                                       voxel_sizes=vox_size, 
                                       origins=[vox_size / 2.] * 3)
            else:
                grid = fvdb.gridbatch_from_ijk(fvdb.JaggedTensor([ijk_tensor]), 
                                       voxel_sizes=vox_size, 
                                       origins=[vox_size / 2.] * 3)
            
            # Free the tensor after grid creation
            del ijk_tensor
            gc.collect()
            if device == 'cuda':
                torch.cuda.empty_cache()
        
            log_memory_usage(f"After grid creation {model_id}")
        except Exception as e:
            print(f"Error in grid creation: {e}")
            clean_memory()
            raise
        
        log_memory_usage(f"After grid creation {model_id}")
        
        # Process reference data efficiently
        try:
            # Use preallocated buffers for tensor creation if available
            if buffers is not None:
                # Get appropriately sized buffers
                point_buffer, normal_buffer = buffers.get_buffers(ref_xyz.shape[0])
                
                # Copy data to buffers efficiently
                with torch.no_grad():
                    # Create tensors on CPU first (less memory pressure)
                    temp_xyz = torch.tensor(ref_xyz, dtype=torch.float32)
                    temp_normal = torch.tensor(ref_normal, dtype=torch.float32)
                    
                    # Then copy to GPU buffers
                    point_buffer.copy_(temp_xyz.to(device))
                    normal_buffer.copy_(temp_normal.to(device))
                    
                    # Free CPU tensors
                    del temp_xyz, temp_normal
                
                ref_xyz_tensor = point_buffer[:ref_xyz.shape[0]]
                ref_normal_tensor = normal_buffer[:ref_normal.shape[0]]
            else:
                # Direct tensor creation on target device
                with torch.no_grad():
                    ref_xyz_tensor = torch.tensor(ref_xyz, dtype=torch.float32, device=device)
                    ref_normal_tensor = torch.tensor(ref_normal, dtype=torch.float32, device=device)
            
            # Free original numpy arrays
            del ref_xyz, ref_normal
            gc.collect()
            
            log_memory_usage(f"After ref data to tensor {model_id}")
        except Exception as e:
            print(f"Error converting reference data to tensors: {e}")
            clean_memory()
            raise
        
        # Process normal data using chunking for large point clouds
        if ref_xyz_tensor.shape[0] > 2_000_000:
            # Define a function to process chunks
            def process_normal_chunk(xyz_chunk, normal_chunk):
                chunk_result = grid.splat_trilinear(fvdb.JaggedTensor(xyz_chunk), 
                                                  fvdb.JaggedTensor(normal_chunk))
                # Normalize
                chunk_result.jdata /= (chunk_result.jdata.norm(dim=1, keepdim=True) + 1e-6)
                return chunk_result
            
            # Process in chunks
            chunks = process_point_cloud_in_chunks(ref_xyz_tensor, ref_normal_tensor, 
                                                  process_normal_chunk, chunk_size=1_000_000)
            
            # Safety check
            if not chunks:
                raise ValueError("Chunking produced no results")
            
            # NOTE: This combination logic depends on the fvdb.JaggedTensor implementation
            # and may need to be adjusted based on the actual API.
            if hasattr(chunks[0], 'jdata'):
                input_normal = chunks[0]  # Starting point
                for chunk in chunks[1:]:
                    if hasattr(chunk, 'jdata'):
                        # Combine chunks (implementation depends on fvdb)
                        try:
                            input_normal.jdata = torch.cat([input_normal.jdata, chunk.jdata], dim=0)
                        except Exception as e:
                            print(f"Warning: Error combining chunks: {e}")
                            print("Using only first chunk")
                            break
                    else:
                        print("Warning: Chunk missing jdata attribute")
                        break
            else:
                print("Warning: First chunk missing jdata attribute")
                input_normal = chunks[0]  # Use as is
        else:
            # Process normally for smaller point clouds
            input_normal = grid.splat_trilinear(fvdb.JaggedTensor(ref_xyz_tensor), 
                                              fvdb.JaggedTensor(ref_normal_tensor))
            # Normalize normal
            input_normal.jdata /= (input_normal.jdata.norm(dim=1, keepdim=True) + 1e-6)
        
        log_memory_usage(f"After normal processing {model_id}")
        
        # Normalize xyz to conv-onet scale
        xyz = grid.grid_to_world(grid.ijk.float()).jdata
        xyz_norm = xyz * 128 / 100
        ref_xyz_norm = ref_xyz_tensor * 128 / 100
        
        # Convert to int32 only for specific resolutions where needed
        if num_vox == 512:
            if xyz_norm.dtype != torch.int32:
                print(f"Converting xyz_norm to int32 for num_vox=512")
                xyz_norm = xyz_norm.to(torch.int32)
        
        try:
            # Convert to fvdb_grid format
            if num_vox == 512:
                # Not splatting
                target_voxel_size = 0.0025
                target_grid = fvdb.gridbatch_from_ijk(
                        fvdb.JaggedTensor(xyz_norm), voxel_sizes=target_voxel_size, origins=[target_voxel_size / 2.] * 3)
            elif num_vox == 16:
                # Splatting
                target_voxel_size = 0.08
                target_grid = fvdb.sparse_grid_from_nearest_voxels_to_points(
                            fvdb.JaggedTensor(xyz_norm), voxel_sizes=target_voxel_size, origins=[target_voxel_size / 2.] * 3)
            elif num_vox == 128:
                # Splatting
                target_voxel_size = 0.01
                target_grid = fvdb.sparse_grid_from_nearest_voxels_to_points(
                            fvdb.JaggedTensor(xyz_norm), voxel_sizes=target_voxel_size, origins=[target_voxel_size / 2.] * 3)
            elif num_vox == 256:
                target_voxel_size = 0.005
                target_grid = fvdb.sparse_grid_from_nearest_voxels_to_points(
                            fvdb.JaggedTensor(xyz_norm), voxel_sizes=target_voxel_size, origins=[target_voxel_size / 2.] * 3)
            elif num_vox == 1024:
                target_voxel_size = 0.00125
                target_grid = fvdb.sparse_grid_from_points(
                            fvdb.JaggedTensor(xyz_norm), voxel_sizes=target_voxel_size, origins=[target_voxel_size / 2.] * 3)
            else:
                raise NotImplementedError(f"Unsupported voxel resolution: {num_vox}")
            
            log_memory_usage(f"After target grid creation {model_id}")
            
        except Exception as e:
            print(f"Error creating target grid for {model_id}: {e}")
            clean_memory()
            return False
        
        # Get target normal with chunking for large point clouds
        if ref_xyz_norm.shape[0] > 2_000_000:
            # Define a function to process chunks
            def process_target_normal_chunk(xyz_chunk, normal_chunk):
                chunk_result = target_grid.splat_trilinear(fvdb.JaggedTensor(xyz_chunk), 
                                                        fvdb.JaggedTensor(normal_chunk))
                # Normalize
                chunk_result.jdata /= (chunk_result.jdata.norm(dim=1, keepdim=True) + 1e-6)
                return chunk_result
            
            # Process in chunks
            chunks = process_point_cloud_in_chunks(ref_xyz_norm, ref_normal_tensor, 
                                                process_target_normal_chunk, chunk_size=1_000_000)
            
            # Safety check
            if not chunks:
                raise ValueError("Target normal chunking produced no results")
            
            # NOTE: This combination logic depends on the fvdb.JaggedTensor implementation
            if hasattr(chunks[0], 'jdata'):
                target_normal = chunks[0]  # Starting point
                for chunk in chunks[1:]:
                    if hasattr(chunk, 'jdata'):
                        # Combine chunks (implementation depends on fvdb)
                        try:
                            target_normal.jdata = torch.cat([target_normal.jdata, chunk.jdata], dim=0)
                        except Exception as e:
                            print(f"Warning: Error combining target normal chunks: {e}")
                            print("Using only first chunk")
                            break
                    else:
                        print("Warning: Target normal chunk missing jdata attribute")
                        break
            else:
                print("Warning: First target normal chunk missing jdata attribute")
                target_normal = chunks[0]  # Use as is
        else:
            # Process normally for smaller point clouds
            target_normal = target_grid.splat_trilinear(fvdb.JaggedTensor(ref_xyz_norm), 
                                                     fvdb.JaggedTensor(ref_normal_tensor))
            target_normal.jdata /= (target_normal.jdata.norm(dim=1, keepdim=True) + 1e-6)
        
        log_memory_usage(f"After target normal calculation {model_id}")
        
        # Move to CPU for saving
        save_dict = {
            "points": target_grid.to("cpu"),
            "normals": target_normal.cpu(),
            "ref_xyz": ref_xyz_norm.cpu(),
            "ref_normal": ref_normal_tensor.cpu(),
        }
        
        log_memory_usage(f"After CPU transfer for saving {model_id}")
        
        # Save with proper checkpointing
        if args.use_checkpointing:
            # Save in a memory-safe way with checkpointing
            try:
                # First save to a temporary file
                temp_path = target_path + '.tmp'
                torch.save(save_dict, temp_path)
                # Then rename to final path (atomic operation)
                if os.path.exists(temp_path):  # Verify file was created
                    os.rename(temp_path, target_path)
                    print(f"Saved with checkpointing: {target_path}")
                else:
                    print(f"Error: Temporary file {temp_path} was not created")
                    return False
            except Exception as e:
                print(f"Error saving {model_id} with checkpointing: {e}")
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception as e2:
                        print(f"Error removing temporary file: {e2}")
                return False
        else:
            # Direct save without checkpointing
            try:
                torch.save(save_dict, target_path)
                print(f"Saved directly: {target_path}")
            except Exception as e:
                print(f"Error saving {model_id}: {e}")
                return False
        
        log_memory_usage(f"After saving {model_id}")
        return True
        
    except Exception as e:
        print(f"Error processing model {model_id}: {e}")
        return False
    finally:
        # Don't clean buffers here as they are shared
        # Just clean temporary objects
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def process_chunk(chunk_ids, category_dir, target_dir, args):
    """Process a chunk of models in a separate process"""
    # Initialize CUDA for this process if available
    if torch.cuda.is_available():
        # Set device to 0 (assuming single GPU)
        torch.cuda.set_device(0)
        print(f"Worker process using CUDA device: {torch.cuda.get_device_name(0)}")
    
    results = []
    # Initialize buffers for this process
    if torch.cuda.is_available():
        local_buffers = ProcessingBuffers(max_points=args.max_sample_points)
    else:
        local_buffers = None
    
    for model_id in chunk_ids:
        model_path = os.path.join(category_dir, model_id)
        try:
            v, f = pcu.load_mesh_vf(model_path)
            success = process_single_model((model_id, v, f), target_dir, args, local_buffers)
            results.append(success)
        except Exception as e:
            print(f"Error loading model {model_id}: {e}")
            results.append(False)
    
    # Clean up buffers
    if local_buffers is not None:
        local_buffers.clear()
    
    return results

def main():
    args = argparse.ArgumentParser()
    args.add_argument('--data_root', type=str, default='./ply_files/')
    args.add_argument('--target_root', type=str, default='./voxels/')
    args.add_argument('--num_vox', type=int, default=512)
    args.add_argument('--categories', type=str, default='000-000')
    args.add_argument('--num_split', type=int, default=8)
    args.add_argument('--split_id', type=int, default=0)
    args.add_argument('--batch_size', type=int, default=3)
    args.add_argument('--max_sample_points', type=int, default=5_000_000)
    args.add_argument('--use_dynamic_sampling', action='store_true', help='Enable dynamic point sampling based on mesh complexity')
    args.add_argument('--use_checkpointing', action='store_true', help='Enable checkpointing to save progress between steps')
    args.add_argument('--parallel_workers', type=int, default=1, help='Number of parallel worker processes')
    args.add_argument('--prefetch_size', type=int, default=2, help='Number of meshes to prefetch')
    args = args.parse_args()

    log_memory_usage("Script start")

    data_root = args.data_root
    target_root = args.target_root

    categories = args.categories.split(',')
    num_vox = args.num_vox

    # Create shared processing buffers if using GPU
    if torch.cuda.is_available():
        buffers = ProcessingBuffers(max_points=args.max_sample_points)
    else:
        buffers = None

    for category in categories:
        category_dir = os.path.join(data_root, category)
        print(category_dir, "category_dir")
        
        # Load model IDs
        model_ids = sorted([f for f in os.listdir(category_dir) if os.path.isfile(os.path.join(category_dir, f)) and f.endswith('.ply')])
        num_models = len(model_ids)
        print("Total number of models in category %s: %d" % (category, num_models))
        
        # Split models based on split_id
        num_models_per_split = num_models // args.num_split
        if args.split_id == args.num_split - 1:
            model_ids = model_ids[args.split_id * num_models_per_split:]
        else:
            model_ids = model_ids[args.split_id * num_models_per_split: (args.split_id + 1) * num_models_per_split]
        
        print(f"Processing {len(model_ids)} models in split {args.split_id} of category {category}")
        target_dir = os.path.join(target_root, "%s" % str(num_vox), category)
        os.makedirs(target_dir, exist_ok=True)
        
        # Filter out models that already have outputs
        filtered_model_ids = []
        for model_id in model_ids:
            target_path = os.path.join(target_dir, "%s.pkl" % model_id.split(".")[0])
            if not os.path.exists(target_path):
                filtered_model_ids.append(model_id)
        
        if len(filtered_model_ids) == 0:
            print(f"All models in category {category} already processed, skipping")
            continue
        
        print(f"Need to process {len(filtered_model_ids)} of {len(model_ids)} models")
        
        # Option 1: Parallel processing with multiple processes
        if args.parallel_workers > 1:
            print(f"Using parallel processing with {args.parallel_workers} workers")
            
            # Split model_ids into chunks for parallel processing
            chunk_size = len(filtered_model_ids) // args.parallel_workers
            if chunk_size == 0:
                chunk_size = 1
            
            chunks = [filtered_model_ids[i:i + chunk_size] for i in range(0, len(filtered_model_ids), chunk_size)]
            
            with ProcessPoolExecutor(max_workers=args.parallel_workers) as executor:
                futures = []
                for chunk in chunks:
                    try:
                        futures.append(executor.submit(
                            process_chunk, chunk, category_dir, target_dir, args))
                    except Exception as e:
                        print(f"Error submitting chunk: {e}")
                
                # Wait for all processes to complete
                for i, future in enumerate(tqdm(futures, desc="Processing chunks")):
                    try:
                        results = future.result()
                        print(f"Chunk {i+1}/{len(futures)} completed: {sum(results)} successful, {len(results) - sum(results)} failed")
                    except Exception as e:
                        print(f"Error processing chunk {i+1}/{len(futures)}: {e}")
        
        # Option 2: Sequential processing with prefetching
        else:
            print(f"Using sequential processing with prefetching (size={args.prefetch_size})")
            
            # Start prefetcher
            prefetcher = MeshPrefetcher(filtered_model_ids, category_dir, prefetch_size=args.prefetch_size)
            
            try:
                # Process models one by one with prefetched data
                processed_count = 0
                success_count = 0
                
                while True:
                    model_id, v, f = prefetcher.get_next()
                    
                    # Check for end of data
                    if model_id is None:
                        break
                    
                    # Process the model
                    success = process_single_model((model_id, v, f), target_dir, args, buffers)
                    processed_count += 1
                    if success:
                        success_count += 1
                    
                    # Periodic status update
                    if processed_count % 10 == 0:
                        print(f"Processed {processed_count}/{len(filtered_model_ids)} models, {success_count} successful")
            
            finally:
                # Stop prefetcher
                prefetcher.stop()
            
            print(f"Processing complete: {success_count}/{processed_count} models successful")
        
        # Clean up shared buffers
        if buffers is not None:
            buffers.clear()
        
        # Final cleanup
        clean_memory()
    
    log_memory_usage("Script end")

if __name__ == "__main__":
    main()