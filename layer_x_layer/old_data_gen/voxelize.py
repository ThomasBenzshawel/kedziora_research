import os
import point_cloud_utils as pcu
import numpy as np
from tqdm import tqdm
import fvdb
import torch
import argparse
import psutil
import gc

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

args = argparse.ArgumentParser()
args.add_argument('--data_root', type=str, default='./ply_files/')
args.add_argument('--target_root', type=str, default='./voxels/')
args.add_argument('--num_vox', type=int, default=512)
args.add_argument('--categories', type=str, default='000-000')
args.add_argument('--num_split', type=int, default=8)
args.add_argument('--split_id', type=int, default=0)
args.add_argument('--batch_size', type=int, default=3)  # Process models in batches
args.add_argument('--max_sample_points', type=int, default=5_000_000)  # Allow limiting sample points
args.add_argument('--use_dynamic_sampling', action='store_true', help='Enable dynamic point sampling based on mesh complexity')
args.add_argument('--use_checkpointing', action='store_true', help='Enable checkpointing to save progress between steps')
args = args.parse_args()

log_memory_usage("Script start")

data_root = args.data_root
target_root = args.target_root

categories = args.categories.split(',')
num_vox = args.num_vox

# Adjust sampling based on resolution
if num_vox > 512:
    max_num_vox = num_vox
    sample_pcs_num = min(args.max_sample_points, 5_000_000)
else:
    max_num_vox = 512
    sample_pcs_num = min(args.max_sample_points, 1_000_000)
vox_size = 1.0 / max_num_vox

log_memory_usage("After initialization")

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
    
    # Process models in batches to manage memory better
    for batch_idx in range(0, len(model_ids), args.batch_size):
        batch_model_ids = model_ids[batch_idx:batch_idx + args.batch_size]
        print(f"Processing batch {batch_idx//args.batch_size + 1}/{(len(model_ids) + args.batch_size - 1)//args.batch_size}")
        
        for model_id in tqdm(batch_model_ids):
            target_path = os.path.join(target_dir, "%s.pkl" %model_id.split(".")[0])
            # check if target_path exists
            if os.path.exists(target_path):
                continue
            
            log_memory_usage(f"Before processing model {model_id}")
            
            try:
                model_path = os.path.join(category_dir, model_id)
                
                # Check for checkpoint if checkpointing is enabled
                checkpoint_path = os.path.join(target_dir, f"{model_id.split('-')[0]}_checkpoint.pkl")
                if args.use_checkpointing and os.path.exists(checkpoint_path):
                    print(f"Found checkpoint for {model_id}, checking stage...")
                    try:
                        checkpoint = torch.load(checkpoint_path)
                        if checkpoint.get("stage") == "before_final_save" and checkpoint.get("model_id") == model_id:
                            print(f"Resuming from final save stage for {model_id}")
                            # Skip to next model since we were about to save or had just saved
                            os.remove(checkpoint_path)  # Clean up checkpoint
                            continue
                        print(f"Checkpoint stage not recognized, starting from beginning")
                    except Exception as e:
                        print(f"Error loading checkpoint: {e}, starting from beginning")
                
                v, f = pcu.load_mesh_vf(os.path.join(model_path))
                log_memory_usage(f"After loading mesh {model_id}")
                
                # Create checkpoint after loading if checkpointing is enabled
                if args.use_checkpointing:
                    try:
                        torch.save({"stage": "after_loading", "model_id": model_id}, checkpoint_path)
                        print(f"Created checkpoint after loading: {checkpoint_path}")
                    except Exception as e:
                        print(f"Failed to create checkpoint after loading: {e}")
                
                # Determine sample size
                if args.use_dynamic_sampling:
                    # Dynamically adjust sample size based on mesh complexity
                    adjusted_sample_size = min(sample_pcs_num, f.shape[0] * 100)
                    print(f"Using dynamic sampling: {adjusted_sample_size} sample points for {model_id}")
                else:
                    # Use fixed sample size
                    adjusted_sample_size = sample_pcs_num
                    print(f"Using fixed sampling: {adjusted_sample_size} sample points for {model_id}")
                
                try:
                    # we need to use a better way of determining adjusted sample size that does not interfere wit the 
                    fid, bc = pcu.sample_mesh_random(v, f, adjusted_sample_size)
                    ref_xyz = pcu.interpolate_barycentric_coords(f, fid, bc, v)
                except Exception as e:
                    print(f"First sampling attempt failed: {e}, retrying with reduced sampling")
                    # Try with reduced sampling regardless of dynamic sampling setting
                    fallback_sample_size = adjusted_sample_size // 2
                    print(f"Fallback to {fallback_sample_size} sample points")
                    fid, bc = pcu.sample_mesh_random(v, f, fallback_sample_size)
                    ref_xyz = pcu.interpolate_barycentric_coords(f, fid, bc, v)
                
                log_memory_usage(f"After sampling {model_id}")
            
                n = pcu.estimate_mesh_face_normals(v, f)
                ref_normal = n[fid]
                log_memory_usage(f"After normal estimation {model_id}")
                
                # Calculate voxelization on CPU
                ijk = pcu.voxelize_triangle_mesh(v, f.astype(np.int32), vox_size, np.zeros(3))
                log_memory_usage(f"After voxelization {model_id}")
                
                # Move to GPU only when necessary
                log_memory_usage(f"Before moving to GPU {model_id}")
                try:
                    # Ensure ijk has the correct integer type
                    # First check if ijk is already an integer type
                    if not np.issubdtype(ijk.dtype, np.integer):
                        print(f"Warning: ijk has non-integer type {ijk.dtype}, converting to int32")
                        ijk = ijk.astype(np.int32)
                    
                    # Convert to tensor with appropriate dtype - keep int32, don't convert
                    ijk_tensor = torch.from_numpy(ijk)
                    
                    # Verify tensor has integer type
                    if ijk_tensor.dtype != torch.int32:
                        print(f"Warning: tensor has non-integral type {ijk_tensor.dtype}, converting to torch.int32")
                        ijk_tensor = ijk_tensor.to(torch.int32)
                    
                    print(f"ijk tensor type: {ijk_tensor.dtype}, shape: {ijk_tensor.shape}")
                    
                    # Free CPU memory immediately after tensor creation
                    del ijk
                    gc.collect()
                    log_memory_usage(f"After creating tensor, before GPU transfer {model_id}")
                    
                    # Now use GPU if available
                    if torch.cuda.is_available():
                        # Check GPU memory before transfer
                        free_gpu_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
                        tensor_size = ijk_tensor.element_size() * ijk_tensor.nelement()
                        
                        if tensor_size > free_gpu_mem * 0.9:  # If tensor would take >90% of free GPU memory
                            print(f"Warning: Tensor size ({tensor_size/1e9:.2f} GB) is close to available GPU memory ({free_gpu_mem/1e9:.2f} GB)")
                            print("Attempting to process in smaller chunks or reduce precision")
                            
                            # Try chunking or processing on CPU if too large
                            if tensor_size > free_gpu_mem * 0.95:
                                print("Not enough GPU memory, processing on CPU instead")
                                is_on_gpu = False
                                grid = fvdb.gridbatch_from_ijk(fvdb.JaggedTensor([ijk_tensor]), 
                                                           voxel_sizes=vox_size, 
                                                           origins=[vox_size / 2.] * 3)
                            else:
                                # Proceed with caution
                                ijk_cuda = ijk_tensor.cuda()
                                is_on_gpu = True
                                grid = fvdb.gridbatch_from_ijk(fvdb.JaggedTensor([ijk_cuda]), 
                                                           voxel_sizes=vox_size, 
                                                           origins=[vox_size / 2.] * 3)
                                # Free the CPU tensor after GPU transfer
                                del ijk_tensor
                        else:
                            # Normal GPU processing
                            ijk_cuda = ijk_tensor.cuda()
                            is_on_gpu = True
                            grid = fvdb.gridbatch_from_ijk(fvdb.JaggedTensor([ijk_cuda]), 
                                                       voxel_sizes=vox_size, 
                                                       origins=[vox_size / 2.] * 3)
                            # Free the CPU tensor after GPU transfer
                            del ijk_tensor
                    else:
                        print("Warning: CUDA not available, using CPU")
                        is_on_gpu = False
                        grid = fvdb.gridbatch_from_ijk(fvdb.JaggedTensor([ijk_tensor]), 
                                                   voxel_sizes=vox_size, 
                                                   origins=[vox_size / 2.] * 3)
                
                    log_memory_usage(f"After grid creation {model_id}")
                except Exception as e:
                    print(f"Error in grid creation: {e}")
                    clean_memory()
                    raise
                
                log_memory_usage(f"After grid creation {model_id}")
                
                # Move reference data to GPU with memory management
                try:
                    # Convert to tensors with memory optimization
                    log_memory_usage(f"Before ref data to tensor {model_id}")
                    
                    # Process in smaller chunks if reference data is large
                    if ref_xyz.shape[0] > 2_000_000:
                        print(f"Large reference data ({ref_xyz.shape[0]} points). Processing in chunks.")
                        # Create tensors with proper memory management
                        if is_on_gpu and torch.cuda.is_available():
                            # Process in chunks
                            chunk_size = 1_000_000
                            ref_xyz_chunks = []
                            ref_normal_chunks = []
                            
                            for i in range(0, ref_xyz.shape[0], chunk_size):
                                end_idx = min(i + chunk_size, ref_xyz.shape[0])
                                # Process each chunk
                                chunk_xyz = torch.from_numpy(ref_xyz[i:end_idx]).float().cuda()
                                chunk_normal = torch.from_numpy(ref_normal[i:end_idx]).float().cuda()
                                ref_xyz_chunks.append(chunk_xyz)
                                ref_normal_chunks.append(chunk_normal)
                                log_memory_usage(f"Processed ref data chunk {i//chunk_size + 1}")
                            
                            # Concatenate chunks
                            ref_xyz_tensor = torch.cat(ref_xyz_chunks, dim=0)
                            ref_normal_tensor = torch.cat(ref_normal_chunks, dim=0)
                            
                            # Clean up intermediate chunks
                            del ref_xyz_chunks, ref_normal_chunks
                            gc.collect()
                            torch.cuda.empty_cache()
                        else:
                            # CPU processing
                            ref_xyz_tensor = torch.from_numpy(ref_xyz).float()
                            ref_normal_tensor = torch.from_numpy(ref_normal).float()
                    else:
                        # Standard processing for smaller data
                        if is_on_gpu and torch.cuda.is_available():
                            ref_xyz_tensor = torch.from_numpy(ref_xyz).float().cuda()
                            ref_normal_tensor = torch.from_numpy(ref_normal).float().cuda()
                        else:
                            ref_xyz_tensor = torch.from_numpy(ref_xyz).float()
                            ref_normal_tensor = torch.from_numpy(ref_normal).float()
                    
                    # Free original numpy arrays
                    del ref_xyz, ref_normal
                    gc.collect()
                    
                    log_memory_usage(f"After ref data to tensor {model_id}")
                except Exception as e:
                    print(f"Error converting reference data to tensors: {e}")
                    clean_memory()
                    raise
                
                # Process normal data
                input_normal = grid.splat_trilinear(fvdb.JaggedTensor(ref_xyz_tensor), 
                                                   fvdb.JaggedTensor(ref_normal_tensor))
                # normalize normal
                input_normal.jdata /= (input_normal.jdata.norm(dim=1, keepdim=True) + 1e-6)  # avoid nan
                
                log_memory_usage(f"After normal processing {model_id}")
                
                # normalize xyz to conv-onet scale
                xyz = grid.grid_to_world(grid.ijk.float()).jdata
                xyz_norm = xyz * 128 / 100
                ref_xyz_norm = ref_xyz_tensor * 128 / 100

                print(f"xyz_norm shape: {xyz_norm.shape}, xyz_norm dtype: {xyz_norm.dtype}")
                

                try:
                    
                    # convert to fvdb_grid format
                    if num_vox == 512:
                        # not splatting
                        if xyz_norm .dtype != torch.int32:
                            print(f"Warning: tensor has non-integral type {xyz_norm .dtype}, converting to torch.int32")
                            xyz_norm  = xyz_norm.to(torch.int32)

                        target_voxel_size = 0.0025
                        target_grid = fvdb.gridbatch_from_ijk(
                                fvdb.JaggedTensor(xyz_norm), voxel_sizes=target_voxel_size, origins=[target_voxel_size / 2.] * 3)
                    elif num_vox == 16:
                        # splatting
                        target_voxel_size = 0.08
                        target_grid = fvdb.sparse_grid_from_nearest_voxels_to_points(
                                    fvdb.JaggedTensor(xyz_norm), voxel_sizes=target_voxel_size, origins=[target_voxel_size / 2.] * 3)
                    elif num_vox == 128:
                        # splatting
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
                        raise NotImplementedError
                    
                    log_memory_usage(f"After target grid creation {model_id}")
                    
                except Exception as e:
                    print(f"Error creating target grid for {model_id}: {e}")
                    clean_memory()
                    continue
                
                # get target normal
                target_normal = target_grid.splat_trilinear(fvdb.JaggedTensor(ref_xyz_norm), fvdb.JaggedTensor(ref_normal_tensor))
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
                
                # Create intermediate checkpoint if enabled
                if args.use_checkpointing:
                    checkpoint_path = os.path.join(target_dir, f"{model_id.split('-')[0]}_checkpoint.pkl")
                    try:
                        torch.save({"stage": "before_final_save", "model_id": model_id}, checkpoint_path)
                        print(f"Created intermediate checkpoint: {checkpoint_path}")
                    except Exception as e:
                        print(f"Failed to create intermediate checkpoint: {e}")
                        # Continue processing even if checkpointing fails
                
                # Save with or without checkpointing
                if args.use_checkpointing:
                    # Save in a memory-safe way with checkpointing
                    try:
                        # First save to a temporary file
                        temp_path = target_path + '.tmp'
                        torch.save(save_dict, temp_path)
                        # Then rename to final path
                        os.rename(temp_path, target_path)
                        print(f"Saved with checkpointing: {target_path}")
                    except Exception as e:
                        print(f"Error saving {model_id} with checkpointing: {e}")
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                else:
                    # Direct save without checkpointing
                    try:
                        torch.save(save_dict, target_path)
                        print(f"Saved directly: {target_path}")
                    except Exception as e:
                        print(f"Error saving {model_id}: {e}")
                
                log_memory_usage(f"After saving {model_id}")
                
            except Exception as e:
                print(f"Error processing model {model_id}: {e}")
            finally:
                # Clean up resources after each model
                clean_memory()
        
        # Extra cleanup after each batch
        print("Completed batch, performing thorough cleanup")
        clean_memory()
        
log_memory_usage("Script end")