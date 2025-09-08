#!/usr/bin/env python3
"""
Parallel worker version of the LLaVA image processing script
Supports distributed processing across multiple GPUs using SLURM
"""

import sys
import os

# Completely disable debugger
sys.settrace(None)
sys.setprofile(None)

# Override all pdb entry points
import pdb
pdb.set_trace = lambda: None
pdb.post_mortem = lambda: None

# Disable any environment-based debugging
os.environ.pop('PYTHONBREAKPOINT', None)
os.environ['PYTHONBREAKPOINT'] = '0'


from time import time
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Set default dtype for Flash Attention compatibility
torch.set_default_dtype(torch.float16)

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from PIL import Image
import requests
import copy
import sys
import warnings
import tqdm
import objaverse
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import time as time_module
import json
import argparse
import fcntl
import hashlib
import random

# Disable any debugger calls
import builtins
builtins.breakpoint = lambda: None
sys.breakpointhook = lambda: None

def get_gpu_id():
    """Get GPU ID from CUDA_VISIBLE_DEVICES or SLURM environment"""
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        gpu_ids = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        return int(gpu_ids[0]) if gpu_ids[0].isdigit() else 0
    elif 'SLURM_LOCALID' in os.environ:
        return int(os.environ['SLURM_LOCALID'])
    else:
        return 0

def load_model_with_flash_attention():
    """Load LLaVA model with Flash Attention enabled"""
    warnings.filterwarnings("ignore")
    
    # Model configuration
    pretrained = "lmms-lab/llava-onevision-qwen2-7b-ov"
    model_name = "llava_qwen"
    
    # Get GPU ID from environment
    gpu_id = get_gpu_id()
    device = f"cuda:{gpu_id}"
    device_map = device
    
    print(f"Worker loading model on {device}")
    
    # Try Flash Attention 2 first
    llava_model_args = {
        "multimodal": True,
        "attn_implementation": None,  # Will be set below
        "torch_dtype": None,  # Will be set below
    }
    
    overwrite_config = {}
    overwrite_config["image_aspect_ratio"] = "pad"
    llava_model_args["overwrite_config"] = overwrite_config
    
    # Try Flash Attention 2 first
    try:
        print("Attempting to load with Flash Attention 2...")
        llava_model_args["attn_implementation"] = "flash_attention_2"
        llava_model_args["torch_dtype"] = torch.bfloat16
        
        tokenizer, model, image_processor, max_length = load_pretrained_model(
            pretrained, None, model_name, 
            device_map=device_map, 
            **llava_model_args
        )
        print("Flash Attention 2 enabled successfully!")
        attention_type = "flash_attention_2"
        
    except Exception as e:
        print(f"Flash Attention 2 failed: {e}")
        print("Falling back to SDPA...")
        
        try:
            # Fallback to SDPA
            llava_model_args["attn_implementation"] = "sdpa"
            llava_model_args["torch_dtype"] = torch.bfloat16
            
            tokenizer, model, image_processor, max_length = load_pretrained_model(
                pretrained, None, model_name, 
                device_map=device_map, 
                **llava_model_args
            )
            print("SDPA attention enabled successfully!")
            attention_type = "sdpa"
            
        except Exception as e2:
            print(f"SDPA failed: {e2}")
            print("Falling back to default attention...")
            
            # Final fallback to default
            llava_model_args["attn_implementation"] = None
            llava_model_args["torch_dtype"] = torch.float16
            
            tokenizer, model, image_processor, max_length = load_pretrained_model(
                pretrained, None, model_name, 
                device_map=device_map, 
                **llava_model_args
            )
            print("Using default attention (slower)")
            attention_type = "default"
    
    # Verify attention implementation
    if hasattr(model.config, 'attn_implementation'):
        print(f"Attention implementation: {model.config.attn_implementation}")
    
    if hasattr(torch, 'compile'):
        print("Compiling model for faster inference...")
        model = torch.compile(model, mode="reduce-overhead")

    model.eval()
    model.to(device)
    
    return tokenizer, model, image_processor, max_length, attention_type, device

def optimized_inference(model, tokenizer, input_ids, images, image_sizes, max_new_tokens=2048):
    """Enhanced inference with better generation parameters"""
    with torch.no_grad():
        dtype = next(model.parameters()).dtype
        
        with torch.autocast(device_type='cuda', dtype=dtype):
            response = model.generate(
                input_ids,
                images=images,
                image_sizes=image_sizes,
                do_sample=True,  # Allow some creativity
                temperature=0.1,  # Very low but not zero for slight variation
                top_p=0.9,  # Nucleus sampling for better quality
                max_new_tokens=max_new_tokens,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.05,  # Reduce repetition
                length_penalty=1.0,  # Neutral length preference
            )
    
    return tokenizer.batch_decode(response, skip_special_tokens=True)[0]

def fetch_object_metadata(folder_path):
    """Fetch metadata for a single object"""
    object_uid = folder_path.split("/")[-1]
    try:
        start_time = time()
        object_metadata = objaverse.load_annotations([object_uid])
        search_time = time() - start_time
        
        if 'name' in object_metadata[object_uid]:
            object_name = object_metadata[object_uid]['name']
        else:
            object_name = "Unknown Object"
        
        return object_uid, object_name, search_time
    except Exception as e:
        print(f"Error fetching metadata for {object_uid}: {e}")
        return object_uid, "Unknown Object", 0

class DistributedMetadataCache:
    """Thread-safe cache for object metadata with file locking for distributed access"""
    def __init__(self, worker_id, cache_file="./objaverse_metadata_cache.json"):
        self.worker_id = worker_id
        self.cache = {}
        self.lock = threading.Lock()
        self.search_times = {}
        self.cache_file = cache_file
        self.worker_cache_file = f"./objaverse_metadata_cache_worker_{worker_id}.json"
        self.load_from_file()
    
    def _lock_file(self, file_handle):
        """Lock a file with timeout"""
        max_attempts = 50
        for attempt in range(max_attempts):
            try:
                fcntl.flock(file_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return True
            except (IOError, OSError):
                time_module.sleep(0.1 + random.random() * 0.1)  # Random jitter
        return False
    
    def _unlock_file(self, file_handle):
        """Unlock a file"""
        try:
            fcntl.flock(file_handle.fileno(), fcntl.LOCK_UN)
        except (IOError, OSError):
            pass
    
    def load_from_file(self):
        """Load existing cache from file with file locking"""
        try:
            # Try to load shared cache first
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    if self._lock_file(f):
                        try:
                            data = json.load(f)
                            self.cache = data.get('cache', {})
                            self.search_times = data.get('search_times', {})
                            print(f"Worker {self.worker_id}: Loaded {len(self.cache)} cached metadata entries from shared cache")
                        finally:
                            self._unlock_file(f)
                    else:
                        print(f"Worker {self.worker_id}: Could not lock shared cache, starting with empty cache")
            
            # Load worker-specific cache
            if os.path.exists(self.worker_cache_file):
                with open(self.worker_cache_file, 'r') as f:
                    data = json.load(f)
                    worker_cache = data.get('cache', {})
                    worker_search_times = data.get('search_times', {})
                    # Merge worker cache with shared cache
                    self.cache.update(worker_cache)
                    self.search_times.update(worker_search_times)
                    print(f"Worker {self.worker_id}: Loaded {len(worker_cache)} entries from worker cache")
            
            print(f"Worker {self.worker_id}: Total cached entries: {len(self.cache)}")
                
        except Exception as e:
            print(f"Worker {self.worker_id}: Error loading metadata cache: {e}")
            self.cache = {}
            self.search_times = {}
    
    def save_to_file(self):
        """Save current cache to worker-specific file"""
        try:
            # Save worker-specific cache
            os.makedirs(os.path.dirname(self.worker_cache_file) if os.path.dirname(self.worker_cache_file) else '.', exist_ok=True)
            
            data = {
                'cache': self.cache,
                'search_times': self.search_times,
                'worker_id': self.worker_id,
                'last_updated': time_module.time()
            }
            
            with open(self.worker_cache_file, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Worker {self.worker_id}: Saved {len(self.cache)} metadata entries to worker cache")
            
        except Exception as e:
            print(f"Worker {self.worker_id}: Error saving metadata cache: {e}")
    
    def set(self, object_uid, object_name, search_time):
        with self.lock:
            self.cache[object_uid] = object_name
            self.search_times[object_uid] = search_time
    
    def get(self, object_uid):
        with self.lock:
            return self.cache.get(object_uid, "Unknown Object"), self.search_times.get(object_uid, 0)
    
    def has(self, object_uid):
        with self.lock:
            return object_uid in self.cache
    
    def get_missing_uids(self, folder_paths):
        """Get list of object UIDs that are not in cache"""
        missing = []
        with self.lock:
            for folder_path in folder_paths:
                object_uid = folder_path.split("/")[-1]
                if object_uid not in self.cache:
                    missing.append(folder_path)
        return missing

def get_worker_folders(target_folder, output_dir, worker_id, total_workers):
    """
    Get folders assigned to this worker using consistent hashing
    """
    all_folders = []
    
    print(f"Worker {worker_id}: Scanning folders...")
    for folder in os.listdir(target_folder):
        folder_path = os.path.join(target_folder, folder)
        
        # Skip if not a directory
        if not os.path.isdir(folder_path):
            continue
        
        # Skip if not exactly 6 images
        try:
            num_images = len(os.listdir(folder_path))
            if num_images != 6:
                continue
        except Exception:
            continue
        
        all_folders.append(folder_path)
    
    # Use consistent hashing to assign folders to workers
    worker_folders = []
    for folder_path in all_folders:
        folder_hash = int(hashlib.md5(folder_path.encode()).hexdigest(), 16)
        if folder_hash % total_workers == worker_id:
            worker_folders.append(folder_path)
    
    # Filter out already processed folders
    folders_to_process = []
    for folder_path in worker_folders:
        folder_name = folder_path.split("/")[-1]
        output_file = os.path.join(output_dir, f"{folder_name}.txt")
        if not os.path.exists(output_file):
            folders_to_process.append(folder_path)
    
    print(f"Worker {worker_id}: Assigned {len(worker_folders)} folders, {len(folders_to_process)} need processing")
    
    # Sort by folder size for better load balancing
    def get_folder_size(folder_path):
        total_size = 0
        try:
            for img_file in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_file)
                if os.path.isfile(img_path):
                    total_size += os.path.getsize(img_path)
        except Exception:
            return float('inf')
        return total_size
    
    folders_to_process.sort(key=get_folder_size)
    return folders_to_process

def load_and_preprocess_images(folder_path, target_max_size=1024):
    """Same as original implementation"""
    image_paths = []
    for image_file in sorted(os.listdir(folder_path)):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_paths.append(os.path.join(folder_path, image_file))
    
    if len(image_paths) != 6:
        raise ValueError(f"Expected 6 images, found {len(image_paths)}")
    
    images = []
    for path in image_paths:
        try:
            img = Image.open(path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            original_size = img.size
            if max(original_size) > target_max_size:
                ratio = target_max_size / max(original_size)
                new_size = tuple(int(dim * ratio) for dim in original_size)
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            min_size = 224
            if min(img.size) < min_size:
                scale = min_size / min(img.size)
                new_size = tuple(int(dim * scale) for dim in img.size)
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            images.append(img)
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            raise
    
    return images

def arrange_images_for_context(images):
    """Same as original implementation"""
    if len(images) == 6:
        try:
            arranged = [images[0], images[2], images[1], images[3], images[4], images[5]]
            return arranged
        except IndexError:
            print("Warning: Could not arrange images, using original order")
            return images
    return images

def batch_process_images(image_list, image_processor, model_config, batch_size=8):
    """Same as original implementation"""
    processed_batches = []
    for i in range(0, len(image_list), batch_size):
        batch = image_list[i:i+batch_size]
        processed_batch = process_images(batch, image_processor, model_config)
        processed_batches.extend(processed_batch)
    return processed_batches

def image_data_augmentation(folder_path, images, metadata_cache, tokenizer, model, image_processor, device, output_dir):
    """Process a single folder - same as original but with passed parameters"""
    object_start_time = time()
    folder_name = folder_path.split("/")[-1]
    output_file = os.path.join(output_dir, f"{folder_name}.txt")
    if os.path.exists(output_file):
        return

    object_uid = folder_path.split("/")[-1]
    object_name, search_time = metadata_cache.get(object_uid)
    images = arrange_images_for_context(images)
    
    all_image_tensors = batch_process_images(images, image_processor, model.config, batch_size=6)
    dtype = next(model.parameters()).dtype
    all_image_tensors = [_image.to(dtype=dtype, device=device) for _image in all_image_tensors]
    
    conv_template = "qwen_1_5"
    
    comprehensive_question = f"""You are an expert 3D object analyst specializing in voxel modeling and spatial reconstruction.

    Your task is to analyze these 6 orthogonal views of a {object_name} and create a comprehensive description for voxel-based 3D reconstruction.

    {DEFAULT_IMAGE_TOKEN} {DEFAULT_IMAGE_TOKEN} {DEFAULT_IMAGE_TOKEN}
    {DEFAULT_IMAGE_TOKEN} {DEFAULT_IMAGE_TOKEN} {DEFAULT_IMAGE_TOKEN}

    ## Analysis Instructions:
    Let's think step-by-step through the spatial structure:

    1. First, identify the primary geometric forms visible across all views
    2. Cross-reference features between adjacent views to understand 3D relationships
    3. Trace how each component connects through the different perspectives
    4. Build a mental 3D model by integrating all visual information

    ## Output Format (provide ALL sections):

    **Label:** {object_name} [Revise if the name doesn't accurately describe what you observe]

    **Spatial Observations:**
    - Front view shows: [key features]
    - Side views reveal: [additional geometry]
    - Top/bottom perspectives indicate: [vertical structure]
    - Cross-view consistency: [how features align across views]

    **Component Hierarchy:**
    PRIMARY: [Main body/structure that forms the core]
    SECONDARY: [Attached or supporting elements]
    DETAILS: [Smaller features, decorative elements]

    **Short Description:** 
    [One precise sentence capturing the object's essential form and function]

    **Long Description:**
    [Comprehensive geometric breakdown for voxel reconstruction. Structure your response as:
    - Overall form: The fundamental 3D shape (e.g., cylindrical base with cubic attachments)
    - Component relationships: How parts connect and their relative positions
    - Proportions: Relative sizes without exact measurements (e.g., "handle is approximately 1/3 the height")
    - Construction sequence: Logical build order from base to details
    - Symmetry and patterns: Any repeating elements or mirror symmetries
    Focus ONLY on geometry and spatial relationships. Exclude colors, textures, materials, or environmental context.]

    **Confidence Rating:** [1-10]
    Justify your rating based on:
    - Clarity of views: How well you can see all aspects
    - Feature consistency: How well features align across views
    - Completeness: Whether any parts are occluded or ambiguous

    ## Critical Requirements:
    ✓ Describe the object as a complete 3D form, not as separate images
    ✓ Use spatial reasoning to infer hidden surfaces
    ✓ Prioritize geometric primitives (cubes, cylinders, spheres, etc.)
    ✓ Maintain consistency - if you see a feature in one view, track it through others
    ✓ Be definitive in your descriptions - you are the expert
    ✗ Do NOT mention "images," "views," or "photos"
    ✗ Do NOT describe backgrounds or lighting
    ✗ Do NOT use uncertain language ("appears to be," "might be")
    ✗ Do NOT mention specific measurements or dimensions

    Remember: Your description will be used by someone who has never seen this object to recreate it in 3D using voxel blocks. Make every detail count for spatial understanding."""

    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], comprehensive_question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    
    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    image_sizes = [img.size for img in images]
    
    final_output = optimized_inference(
        model, tokenizer, input_ids, all_image_tensors, image_sizes, max_new_tokens=2048
    )
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(output_file, "w") as f:
        f.write(final_output)
    
    time_taken = time() - object_start_time
    return time_taken

def main():
    parser = argparse.ArgumentParser(description='Parallel LLaVA processing worker')
    parser.add_argument('--worker_id', type=int, required=True, help='Worker ID (0-indexed)')
    parser.add_argument('--total_workers', type=int, required=True, help='Total number of workers')
    parser.add_argument('--target_folder', type=str, default='../objaverse_images', 
                       help='Path to input folder')
    parser.add_argument('--output_dir', type=str, default='../objaverse_descriptions',
                       help='Path to output directory for descriptions')
    parser.add_argument('--save_interval', type=int, default=100, 
                       help='Save cache every N processed items')
    
    args = parser.parse_args()
    
    print(f"Worker {args.worker_id}/{args.total_workers} starting...")
    
    # Load model
    print("Loading LLaVA model with Flash Attention optimization...")
    tokenizer, model, image_processor, max_length, attention_type, device = load_model_with_flash_attention()
    print(f"Model loaded successfully with {attention_type} attention on {device}!")
    
    # Get folders for this worker
    folders_to_process = get_worker_folders(args.target_folder, args.output_dir, args.worker_id, args.total_workers)
    print(f"Worker {args.worker_id}: Processing {len(folders_to_process)} folders...")
    
    if len(folders_to_process) == 0:
        print(f"Worker {args.worker_id}: No folders to process, exiting.")
        return
    
    # Create metadata cache
    metadata_cache = DistributedMetadataCache(args.worker_id)
    
    # Find folders that need metadata fetching
    folders_needing_metadata = metadata_cache.get_missing_uids(folders_to_process)
    
    if len(folders_needing_metadata) > 0:
        print(f"Worker {args.worker_id}: Pre-fetching metadata for {len(folders_needing_metadata)} objects...")
        prefetch_start_time = time()
        
        with ThreadPoolExecutor(max_workers=16) as executor:
            future_to_folder = {executor.submit(fetch_object_metadata, folder_path): folder_path
                              for folder_path in folders_needing_metadata}
            
            completed = 0
            with tqdm.tqdm(total=len(folders_needing_metadata), 
                          desc=f"Worker {args.worker_id} fetching metadata") as pbar:
                for future in as_completed(future_to_folder):
                    folder_path = future_to_folder[future]
                    try:
                        object_uid, object_name, search_time = future.result()
                        metadata_cache.set(object_uid, object_name, search_time)
                    except Exception as e:
                        print(f"Error fetching metadata for {folder_path}: {e}")
                        object_uid = folder_path.split("/")[-1]
                        metadata_cache.set(object_uid, "Unknown Object", 0)
                    
                    completed += 1
                    pbar.update(1)
                    
                    if completed % 200 == 0:
                        metadata_cache.save_to_file()
        
        metadata_cache.save_to_file()
        prefetch_time = time() - prefetch_start_time
        print(f"Worker {args.worker_id}: Metadata prefetch completed in {prefetch_time:.2f}s")
    
    # Process folders
    total_start_time = time()
    successful_processes = 0
    
    print(f"Worker {args.worker_id}: Starting processing...")
    
    with tqdm.tqdm(total=len(folders_to_process), 
                  desc=f"Worker {args.worker_id} processing") as pbar:
        for i, folder_path in enumerate(folders_to_process):
            try:
                images = load_and_preprocess_images(folder_path)
                process_time = image_data_augmentation(
                    folder_path, images, metadata_cache, tokenizer, model, image_processor, device, args.output_dir
                )
                successful_processes += 1
                
                # Memory optimization
                if successful_processes % 10 == 0:
                    torch.cuda.empty_cache()
                
                # Save cache periodically
                if successful_processes % args.save_interval == 0:
                    metadata_cache.save_to_file()
                
            except Exception as e:
                print(f"Worker {args.worker_id}: Error processing {folder_path}: {e}")
                continue
            
            pbar.update(1)
    
    # Final cleanup
    metadata_cache.save_to_file()
    torch.cuda.empty_cache()
    
    total_time = time() - total_start_time
    print(f"\nWorker {args.worker_id} COMPLETE!")
    print(f"Processed: {successful_processes}/{len(folders_to_process)} folders")
    print(f"Total time: {total_time:.2f} seconds")
    if successful_processes > 0:
        print(f"Average time per object: {total_time / successful_processes:.2f} seconds")

if __name__ == "__main__":
    main()