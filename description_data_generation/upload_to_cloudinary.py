import os
import argparse
from glob import glob
import time
from tqdm import tqdm
import logging
from dotenv import load_dotenv
import cloudinary
import cloudinary.uploader
import cloudinary.api

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("image_upload.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Cloudinary Configuration
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME", ""),
    api_key=os.getenv("CLOUDINARY_API_KEY", ""),
    api_secret=os.getenv("CLOUDINARY_API_SECRET", "")
)

def upload_images_for_object(object_id, image_paths, angles=None):
    """Upload multiple images for a specific object to Cloudinary"""
    
    if not angles:
        # If angles aren't specified, try to infer from filenames (e.g., "front_view.jpg" → "front")
        angles = []
        for img_path in image_paths:
            filename = os.path.basename(img_path)
            # Try to extract angle from filename (customize based on your naming convention)
            if "_" in filename:
                angle = filename.split("_")[0]
            else:
                angle = "unspecified"
            angles.append(angle)
    
    results = []
    
    for i, img_path in enumerate(image_paths):
        angle = angles[i] if i < len(angles) else "unspecified"
        
        try:
            # Upload to Cloudinary with object_id and angle as tags
            upload_result = cloudinary.uploader.upload(
                img_path,
                folder=f"objects/{object_id}",
                tags=[object_id, angle],
                public_id=f"{angle}_{os.path.splitext(os.path.basename(img_path))[0]}"
            )
            
            # Store the upload result
            results.append({
                "public_id": upload_result.get("public_id"),
                "url": upload_result.get("secure_url"),
                "angle": angle,
                "original_filename": os.path.basename(img_path)
            })
            
            logger.info(f"Uploaded {os.path.basename(img_path)} for object {object_id} with angle {angle}")
            
        except Exception as e:
            logger.error(f"Error uploading {os.path.basename(img_path)} for object {object_id}: {str(e)}")
    
    success = len(results) > 0
    
    return {
        "success": success,
        "data": results
    }

def batch_upload_from_directory(base_dir, delay=0.5):
    """
    Upload images from a directory structure to Cloudinary.
    Assumes the directory structure is:
    base_dir/
        object_id_1/
            image1.jpg
            image2.jpg
        object_id_2/
            image1.jpg
            ...
    """
    # Find all subdirectories (each should be an object_id)
    object_dirs = [d for d in os.listdir(base_dir) 
                  if os.path.isdir(os.path.join(base_dir, d))]
    
    logger.info(f"Found {len(object_dirs)} object directories to process")
    
    results = {
        "success": 0,
        "failed": 0,
        "skipped": 0
    }
    
    for object_id in tqdm(object_dirs, desc="Uploading objects"):
        object_path = os.path.join(base_dir, object_id)
        
        # Find all images in this directory
        image_files = []
        for ext in ['jpg', 'jpeg', 'png', 'gif']:
            image_files.extend(glob(os.path.join(object_path, f"*.{ext}")))
            image_files.extend(glob(os.path.join(object_path, f"*.{ext.upper()}")))
        
        if not image_files:
            logger.warning(f"No images found for object {object_id}, skipping")
            results["skipped"] += 1
            continue
        
        # Upload the images
        upload_result = upload_images_for_object(object_id, image_files)
        
        if upload_result and upload_result.get("success", False):
            results["success"] += 1
        else:
            results["failed"] += 1
        
        # Add delay to avoid overwhelming the server
        if delay > 0:
            time.sleep(delay)
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch upload images to Cloudinary")
    parser.add_argument("--dir", required=True, help="Base directory containing object subdirectories")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between object uploads (seconds)")
    
    args = parser.parse_args()
    
    print(f"Starting batch upload from directory: {args.dir}")
    results = batch_upload_from_directory(args.dir, args.delay)
    
    print("\nUpload Summary:")
    print(f"  Success: {results['success']} objects")
    print(f"  Failed: {results['failed']} objects")
    print(f"  Skipped: {results['skipped']} objects (no images found)")
    print("\nSee image_upload.log for details")