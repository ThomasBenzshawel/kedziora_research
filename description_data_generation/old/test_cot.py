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
import os
import tqdm
import objaverse
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import time as time_module
import json

# Disable any debugger calls
import builtins
builtins.breakpoint = lambda: None
sys.breakpointhook = lambda: None

def load_model_with_flash_attention():
    """Load LLaVA model with Flash Attention enabled"""
    warnings.filterwarnings("ignore")
    
    # Model configuration
    pretrained = "lmms-lab/llava-onevision-qwen2-7b-ov"
    model_name = "llava_qwen"
    device = "cuda:0"
    device_map = "cuda:0"
    
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
    
    return tokenizer, model, image_processor, max_length, attention_type

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

class MetadataCache:
    """Thread-safe cache for object metadata with file persistence"""
    def __init__(self, cache_file="./objaverse_metadata_cache.json"):
        self.cache = {}
        self.lock = threading.Lock()
        self.search_times = {}
        self.cache_file = cache_file
        self.load_from_file()
    
    def load_from_file(self):
        """Load existing cache from file if it exists"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    self.cache = data.get('cache', {})
                    self.search_times = data.get('search_times', {})
                print(f"Loaded {len(self.cache)} cached metadata entries from {self.cache_file}")
            else:
                print("No existing metadata cache found, starting fresh")
        except Exception as e:
            print(f"Error loading metadata cache: {e}")
            self.cache = {}
            self.search_times = {}
    
    def save_to_file(self):
        """Save current cache to file"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.cache_file) if os.path.dirname(self.cache_file) else '.', exist_ok=True)
            
            data = {
                'cache': self.cache,
                'search_times': self.search_times,
                'last_updated': time_module.time()
            }
            
            with open(self.cache_file, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Saved {len(self.cache)} metadata entries to {self.cache_file}")
        except Exception as e:
            print(f"Error saving metadata cache: {e}")
    
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

def get_processable_folders(target_folder):
    """
    Get folders that need processing, sorted by size (smaller first for faster feedback)
    """
    folders_to_process = []
    
    print("Scanning folders and filtering...")
    for folder in tqdm.tqdm(os.listdir(target_folder), desc="Scanning folders"):
        folder_path = os.path.join(target_folder, folder)
        
        # Skip if not a directory
        if not os.path.isdir(folder_path):
            continue
            
        # Skip if output already exists
        output_file = f"./objaverse_descriptions/{folder}.txt"
        if os.path.exists(output_file):
            continue
            
        # Skip if not exactly 6 images
        try:
            num_images = len(os.listdir(folder_path))
            if num_images != 6:
                print(f"Skipping folder with {num_images} images: {folder_path}")
                continue
        except Exception as e:
            print(f"Error checking folder {folder_path}: {e}")
            continue
            
        folders_to_process.append(folder_path)
    
    print(f"Found {len(folders_to_process)} folders to process")
    
    # Sort by total image size (smaller first for faster initial results and better memory management)
    def get_folder_size(folder_path):
        total_size = 0
        try:
            for img_file in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_file)
                if os.path.isfile(img_path):
                    total_size += os.path.getsize(img_path)
        except Exception:
            # If we can't get size, put it at the end
            return float('inf')
        return total_size
    
    print("Sorting folders by size (smallest first)...")
    folders_to_process.sort(key=get_folder_size)
    
    return folders_to_process

def load_and_preprocess_images(folder_path, target_max_size=1024):
    image_paths = []
    for image_file in sorted(os.listdir(folder_path)):  # Sort for consistent ordering
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_paths.append(os.path.join(folder_path, image_file))
    
    if len(image_paths) != 6:
        raise ValueError(f"Expected 6 images, found {len(image_paths)}")
    
    images = []
    for path in image_paths:
        try:
            img = Image.open(path)
            
            # Convert to RGB if needed (handles RGBA, grayscale, etc.)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Smart resizing - maintain aspect ratio but ensure minimum quality
            original_size = img.size
            if max(original_size) > target_max_size:
                # Calculate new size maintaining aspect ratio
                ratio = target_max_size / max(original_size)
                new_size = tuple(int(dim * ratio) for dim in original_size)
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            # Ensure minimum size for model quality
            min_size = 224  # Most vision models perform better with at least 224x224
            if min(img.size) < min_size:
                # Scale up smaller dimension to maintain aspect ratio
                scale = min_size / min(img.size)
                new_size = tuple(int(dim * scale) for dim in img.size)
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            images.append(img)
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            raise
    
    return images

def arrange_images_for_context(images):
    """
    Arrange images in logical viewing order for better model understanding
    Assumes standard 6-view arrangement: front, back, left, right, top, bottom
    """
    if len(images) == 6:
        # Create a logical viewing sequence: front -> left -> back -> right -> top -> bottom
        # This helps the model understand spatial relationships better
        try:
            arranged = [images[0], images[2], images[1], images[3], images[4], images[5]]
            return arranged
        except IndexError:
            # Fallback to original order if arrangement fails
            print("Warning: Could not arrange images, using original order")
            return images
    return images

def batch_process_images(image_list, image_processor, model_config, batch_size=8):
    """
    Process images in smaller batches for memory efficiency
    """
    processed_batches = []
    for i in range(0, len(image_list), batch_size):
        batch = image_list[i:i+batch_size]
        processed_batch = process_images(batch, image_processor, model_config)
        processed_batches.extend(processed_batch)
    return processed_batches

class AsyncImageLoader:
    """
    Preload images asynchronously while processing previous objects
    """
    def __init__(self, folder_paths, num_workers=2):
        self.folder_paths = folder_paths
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.futures = {}
        self.current_index = 0
        
        # Start preloading first few images
        self._preload_next_batch()
    
    def _preload_next_batch(self, batch_size=3):
        """Preload next batch of images"""
        for i in range(batch_size):
            idx = self.current_index + i
            if idx < len(self.folder_paths) and idx not in self.futures:
                folder_path = self.folder_paths[idx]
                future = self.executor.submit(load_and_preprocess_images, folder_path)
                self.futures[idx] = future
    
    def get_next_images(self):
        """Get next preprocessed images, blocking if not ready"""
        if self.current_index >= len(self.folder_paths):
            return None, None
        
        folder_path = self.folder_paths[self.current_index]
        
        # Get preloaded images
        if self.current_index in self.futures:
            future = self.futures.pop(self.current_index)
            try:
                images = future.result(timeout=30)  # 30 second timeout
            except Exception as e:
                print(f"Error getting preloaded images for {folder_path}: {e}")
                # Fallback to synchronous loading
                images = load_and_preprocess_images(folder_path)
        else:
            # Fallback to synchronous loading
            images = load_and_preprocess_images(folder_path)
        
        self.current_index += 1
        
        # Preload more images
        self._preload_next_batch()
        
        return folder_path, images
    
    def cleanup(self):
        """Cleanup executor"""
        self.executor.shutdown(wait=False)

def image_data_augmentation(folder_path, images, metadata_cache):
    """
    Optimized function with enhanced image arrangement and processing
    """
    object_start_time = time()
    folder_name = folder_path.split("/")[-1]
    output_file = f"./objaverse_descriptions/{folder_name}.txt"
    if os.path.exists(output_file):
        print(f"Output file already exists: {output_file}")
        return

    # Get the uid of the object from the folder name
    object_uid = folder_path.split("/")[-1]

    # Get metadata from cache (should already be available)
    object_name, search_time = metadata_cache.get(object_uid)

    # Arrange images for better context (Recommendation 1)
    images = arrange_images_for_context(images)

    # Process images with batch processing for memory efficiency (Recommendation 6)
    all_image_tensors = batch_process_images(images, image_processor, model.config, batch_size=6)
    
    # Use appropriate dtype based on model
    dtype = next(model.parameters()).dtype
    all_image_tensors = [_image.to(dtype=dtype, device=device) for _image in all_image_tensors]
    
    conv_template = "qwen_1_5"
    
    # Enhanced comprehensive analysis prompt
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
    
    # Use enhanced optimized inference (Recommendation 4)
    final_output = optimized_inference(
        model, tokenizer, input_ids, all_image_tensors, image_sizes, max_new_tokens=2048
    )
    
    # Save output to txt file
    os.makedirs("./objaverse_descriptions", exist_ok=True)
    
    with open(output_file, "w") as f:
        f.write(final_output)
    print(f"Output saved to {output_file}")
    time_taken = time() - object_start_time
    print(f"Time taken for {folder_name}: {time_taken:.2f} seconds")

if __name__ == "__main__":
    print("Loading LLaVA model with Flash Attention optimization...")
    
    # Load model with Flash Attention (same interface as original)
    tokenizer, model, image_processor, max_length, attention_type = load_model_with_flash_attention()
    
    # Make these global so image_data_augmentation can access them (same as original)
    globals()['tokenizer'] = tokenizer
    globals()['model'] = model
    globals()['image_processor'] = image_processor
    globals()['device'] = "cuda:0"
    
    print(f"Model loaded successfully with {attention_type} attention!")
    
    # Get folders to process with better filtering and sorting
    target_folder = "./objaverse_parallel/objaverse_images"
    folders_to_process = get_processable_folders(target_folder)

    print(f"Processing {len(folders_to_process)} folders...")
    
    # Create metadata cache
    metadata_cache = MetadataCache()
    
    # Find folders that need metadata fetching
    folders_needing_metadata = metadata_cache.get_missing_uids(folders_to_process)
        
    if len(folders_needing_metadata) > 0:
        print(f"Pre-fetching metadata for {len(folders_needing_metadata)} objects...")
        prefetch_start_time = time()
        
        # More workers + less frequent saves
        with ThreadPoolExecutor(max_workers=32) as executor:
            future_to_folder = {executor.submit(fetch_object_metadata, folder_path): folder_path
                            for folder_path in folders_needing_metadata}
            
            completed = 0
            with tqdm.tqdm(total=len(folders_needing_metadata), desc="Fetching metadata") as pbar:
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
                    
                    # Save less frequently - every 500 items
                    if completed % 500 == 0:
                        metadata_cache.save_to_file()
        
        # Save once at the end
        metadata_cache.save_to_file()
        
        prefetch_time = time() - prefetch_start_time
        print(f"Completed in {prefetch_time:.2f}s ({len(folders_needing_metadata)/prefetch_time:.1f} items/sec)")
        
    # Process folders with async image loading and memory optimizations
    total_start_time = time()
    successful_processes = 0
    
    print("Starting processing with async image loading...")
    image_loader = AsyncImageLoader(folders_to_process, num_workers=4)
    
    try:
        with tqdm.tqdm(total=len(folders_to_process), desc="Processing with model") as pbar:
            while True:
                folder_path, images = image_loader.get_next_images()
                if folder_path is None:
                    break

                try:
                    image_data_augmentation(folder_path, images, metadata_cache)
                    successful_processes += 1
                    
                    # Memory optimization: Clear GPU cache periodically (Recommendation 6)
                    if successful_processes % 10 == 0:
                        torch.cuda.empty_cache()
                        print(f"GPU cache cleared at {successful_processes} processes")
                    
                except Exception as e:
                    print(f"Error processing {folder_path}: {e}")
                    continue
                
                pbar.update(1)
                
    finally:
        # Clean up the async image loader
        image_loader.cleanup()
        # Final GPU cache cleanup
        torch.cuda.empty_cache()
    
    total_time = time() - total_start_time
    print("\n" + "="*50)
    print("PROCESSING COMPLETE!")
    print("="*50)
    print(f"Total folders processed: {successful_processes}/{len(folders_to_process)}")
    print(f"Total processing time: {total_time:.2f} seconds")
    if successful_processes > 0:
        print(f"Average time per object: {total_time / successful_processes:.2f} seconds")
    if prefetch_time > 0:
        print(f"Metadata pre-fetch time: {prefetch_time:.2f} seconds")
        print(f"New metadata fetched: {len(folders_needing_metadata)} objects")
        print(f"Used cached metadata: {len(folders_to_process) - len(folders_needing_metadata)} objects")
        print(f"Time saved by caching: ~{(len(folders_to_process) - len(folders_needing_metadata)) * 1.0:.2f} seconds")
    else:
        print("Used cached metadata for all objects!")
        print(f"Time saved by caching: ~{len(folders_to_process) * 1.0:.2f} seconds")