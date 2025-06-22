import os
import argparse
from glob import glob
import time
from tqdm import tqdm
import logging
from dotenv import load_dotenv
import requests
import mimetypes
import warnings
warnings.filterwarnings('ignore', message='Unverified HTTPS request')

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

# Get API endpoint from environment variables
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:3000")

def upload_images_for_object(object_id, image_paths, angles=None):
    """Upload multiple images for a specific object using the API endpoint"""
    
    if not angles:
        # If angles aren't specified, try to infer from filenames (e.g., "front_view.jpg" → "front")
        angles = []
        for img_path in image_paths:
            filename = os.path.basename(img_path)
            # Try to extract angle from filename (customize based on your naming convention)
            if "_" in filename:
                angle = filename.split("_")[-1].split(".")[0]  # Get the last part before the extension
            else:
                angle = "unspecified"
            angles.append(angle)
    
    results = []
    
    for i, img_path in enumerate(image_paths):
        angle = angles[i] if i < len(angles) else "unspecified"
        
        try:
            # Prepare the API endpoint URL
            upload_url = f"{API_BASE_URL}/api/objects/{object_id}/images"
            
            # Determine the content type based on file extension
            content_type, _ = mimetypes.guess_type(img_path)
            if not content_type:
                content_type = 'application/octet-stream'
            
            # Open the image file
            with open(img_path, 'rb') as img_file:
                # Create the form data with the file and angle
                files = {'file': (os.path.basename(img_path), img_file, content_type)}
                data = {'angle': angle}
                
                # Make the POST request to the API endpoint
                response = requests.post(upload_url, files=files, data=data, verify=False)
                
                # Check if the request was successful
                if response.status_code == 200:
                    # Parse the response JSON
                    upload_result = response.json()
                    
                    # Store the result if successful
                    if upload_result and upload_result.get('success', False):
                        image_data = upload_result.get('data', {})
                        
                        results.append({
                            "public_id": image_data.get('imageId'),
                            "url": image_data.get('url'),
                            "angle": angle,
                            "original_filename": os.path.basename(img_path)
                        })
                        
                        logger.info(f"Uploaded {os.path.basename(img_path)} for object {object_id} with angle {angle}")
                    else:
                        logger.error(f"API returned success=False for {os.path.basename(img_path)}: {upload_result}")
                else:
                    logger.error(f"API returned status code {response.status_code} for {os.path.basename(img_path)}: {response.text}")
            
        except Exception as e:
            logger.error(f"Error uploading {os.path.basename(img_path)} for object {object_id}: {str(e)}")
    
    success = len(results) > 0
    
    return {
        "success": success,
        "data": results
    }

def batch_upload_from_directory(base_dir, delay=0.5):
    """
    Upload images from a directory structure to the API.
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
        for ext in ['jpg', 'jpeg']:
            image_files.extend(glob(os.path.join(object_path, f"*.{ext}")))
            image_files.extend(glob(os.path.join(object_path, f"*.{ext.upper()}")))
        
        if len(image_files) < 6:
            logger.warning(f"Not enough images found for object {object_id}, skipping")
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
    parser = argparse.ArgumentParser(description="Batch upload images to API")
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