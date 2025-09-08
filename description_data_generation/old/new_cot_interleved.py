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
    """Load LLaVA model with Flash Attention enabled - same interface as original"""
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
        model = torch.compile(model, mode="max-autotune")

    model.eval()
    model.to(device)
    
    return tokenizer, model, image_processor, max_length, attention_type

def optimized_inference(model, tokenizer, input_ids, images, image_sizes, max_new_tokens=1536):
    """Optimized inference with Flash Attention support"""
    with torch.no_grad():
        # Use appropriate dtype based on model
        dtype = next(model.parameters()).dtype
        
        with torch.autocast(device_type='cuda', dtype=dtype):
            response = model.generate(
                input_ids,
                images=images,
                image_sizes=image_sizes,
                do_sample=False,
                temperature=0,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
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
    OPTIMIZATION 6: Get folders that need processing, sorted by size (smaller first for faster feedback)
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
    """
    OPTIMIZATION 8: More efficient image loading and preprocessing
    """
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
            
            # Resize if very large to save memory and processing time
            if img.size[0] > target_max_size or img.size[1] > target_max_size:
                # Use high-quality resampling but limit size
                img.thumbnail((target_max_size, target_max_size), Image.Resampling.LANCZOS)
            
            images.append(img)
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            raise
    
    return images

class AsyncImageLoader:
    """
    OPTIMIZATION 3: Preload images asynchronously while processing previous objects
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
    Optimized function that uses pre-fetched metadata from cache and preloaded images
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

    # Process all images for later use
    all_image_tensors = process_images(images, image_processor, model.config)
    
    # Use appropriate dtype based on model
    dtype = next(model.parameters()).dtype
    all_image_tensors = [_image.to(dtype=dtype, device=device) for _image in all_image_tensors]
    
    conv_template = "qwen_1_5"
    
    # Single comprehensive analysis with all images - optimized prompt
    comprehensive_question = f"""You are an expert object analyst with exceptional attention to detail.
      Analyze this {object_name} from all six angles shown and provide a comprehensive description optimized for voxel modeling.

{DEFAULT_IMAGE_TOKEN} 
{DEFAULT_IMAGE_TOKEN}
{DEFAULT_IMAGE_TOKEN}
{DEFAULT_IMAGE_TOKEN}
{DEFAULT_IMAGE_TOKEN}
{DEFAULT_IMAGE_TOKEN}

Provide your analysis in this exact format:

Label: {object_name} (You may relabel it if the label is not descriptive enough or you have a better name for it)

Short Description: [One sentence capturing the object's essence]

Component List: [List all major components, parts, or features of the object.]

Long Description: [Detailed description focusing on geometric shapes, component relationships,
 and structural details for re-creating the object out of legos. Focus on basic geometric shapes and forms, component relationships,
 distinctive features and their purpose, and construction/connections between parts.
 NEVER EVER mention specific measurements, detailed texture information, material properties like weight/density/glossiness, references to environment/context, and lighting/shadows.
 These are not relevant for voxel modeling and will not help in re-creating the object.]

Confidence Rating: [Rate 1-10 your confidence in the description being accurate and complete. 
1 means you are not confident at all, and 10 means you are very confident that your description is high enough
 quality to model the object having never seen it.]

Important guidelines:
- Do NOT mention the background, environment, or context
- Do NOT refer to images, photos, angles, or views
- Be definitive, you are an expert creating a description for a person who will model this object
- Do NOT mention that this is a 3D object or model, thats a given
- Do NOT mention any specific measurements, dimensions, or sizes
- Describe the object as if it is right in front of you

Here is the object you need to analyze:
{DEFAULT_IMAGE_TOKEN} 
{DEFAULT_IMAGE_TOKEN}
{DEFAULT_IMAGE_TOKEN}
{DEFAULT_IMAGE_TOKEN}
{DEFAULT_IMAGE_TOKEN}
{DEFAULT_IMAGE_TOKEN}

"""

    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], comprehensive_question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    
    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    image_sizes = [img.size for img in images]
    
    # Use optimized inference
    final_output = optimized_inference(
        model, tokenizer, input_ids, all_image_tensors, image_sizes, max_new_tokens=1536
    )
    
    # Save output to txt file
    os.makedirs("./objaverse_descriptions", exist_ok=True)
    
    with open(output_file, "w") as f:
        f.write(final_output)
    print(f"Output saved to {output_file}")
    time_taken = time() - object_start_time
    print(f"Time taken for {folder_name}: {time_taken:.2f} seconds")
    # print the percent of time taken that was spent on the object search
   
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
    
    # OPTIMIZATION 6: Get folders to process with better filtering and sorting
    target_folder = "./objaverse_images"
    folders_to_process = get_processable_folders(target_folder)

    print(f"Processing {len(folders_to_process)} folders...")
    
    # Create metadata cache
    metadata_cache = MetadataCache()
    
    # Find folders that need metadata fetching
    folders_needing_metadata = metadata_cache.get_missing_uids(folders_to_process)
    
    if len(folders_needing_metadata) > 0:
        # Pre-fetch missing metadata in parallel
        print(f"Pre-fetching metadata for {len(folders_needing_metadata)} objects (already have {len(folders_to_process) - len(folders_needing_metadata)} cached)...")
        prefetch_start_time = time()
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            # Submit all metadata fetch jobs
            future_to_folder = {executor.submit(fetch_object_metadata, folder_path): folder_path 
                               for folder_path in folders_needing_metadata}
            
            # Collect results with progress bar
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
                    
                    # Save cache periodically (every 50 items) in case of interruption
                    if completed % 50 == 0:
                        metadata_cache.save_to_file()
        
        prefetch_time = time() - prefetch_start_time
        print(f"Metadata pre-fetching completed in {prefetch_time:.2f} seconds")
        print(f"Average search time per new object: {prefetch_time / len(folders_needing_metadata):.2f} seconds")
        
        # Save final cache
        metadata_cache.save_to_file()
    else:
        print("All metadata already cached! Skipping fetch phase.")
        prefetch_time = 0
    
    # OPTIMIZATION 3: Process folders with async image loading
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
                
                # Duplicate the images, shuffle the duplicates, and then append them to the original list
                dupe_images = images.copy()  # Create a copy to avoid modifying original list
                dupe_images = dupe_images[::-1]  # Reverse the order for variety
                images.extend(dupe_images)  # Append duplicates to the original list

                try:
                    image_data_augmentation(folder_path, images, metadata_cache)
                    successful_processes += 1
                    
                    # print the memory usage
                    # print(f"Memory usage: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
                    # print(f"GPU memory usage: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
                    
                except Exception as e:
                    print(f"Error processing {folder_path}: {e}")
                    continue
                
                pbar.update(1)
                
    finally:
        # Clean up the async image loader
        image_loader.cleanup()
    
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