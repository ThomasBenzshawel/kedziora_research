import os
import cloudinary
import cloudinary.uploader
import cloudinary.api
import json
import time
from tqdm import tqdm
import pandas as pd
from pymongo import MongoClient

# Cloudinary configuration
cloudinary.config( 
    cloud_name = "dj0gvopya", 
    api_key = "316816699942876", 
    api_secret = "h9-WpLDzxJNjlOx0ldlwSjQLytk",
    secure = True
)

# Paths to your data
IMAGES_FOLDER = "/path/to/images_folder"
DESCRIPTIONS_FOLDER = "/path/to/descriptions_folder"

# MongoDB connection
MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "image_description_eval"
client = MongoClient(MONGO_URI)
db = client[DB_NAME]

def upload_images_to_cloudinary(folder_path, batch_size=100):
    """Upload images from a folder to Cloudinary with rate limiting and batching"""
    uploaded_images = []
    image_files = [f for f in os.listdir(folder_path) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp'))]
    
    print(f"Found {len(image_files)} images to upload")
    
    # Process in batches to avoid rate limits
    for i in tqdm(range(0, len(image_files), batch_size)):
        batch = image_files[i:i+batch_size]
        
        for img_file in batch:
            # Extract ID from filename (assuming filename is ID.extension)
            img_id = os.path.splitext(img_file)[0]
            img_path = os.path.join(folder_path, img_file)
            
            try:
                # Upload to Cloudinary with the ID as public_id
                result = cloudinary.uploader.upload(
                    img_path,
                    public_id=img_id,
                    folder="evaluation_images",  # Organize in a folder
                    resource_type="image",
                    overwrite=True
                )
                
                uploaded_images.append({
                    "image_id": img_id,
                    "image_url": result["secure_url"],
                    "cloudinary_public_id": result["public_id"]
                })
                
            except Exception as e:
                print(f"Error uploading {img_file}: {str(e)}")
        
        # Avoid rate limits - pause between batches
        if i + batch_size < len(image_files):
            time.sleep(2)
    
    # Save upload results to a file for backup
    with open("cloudinary_upload_results.json", "w") as f:
        json.dump(uploaded_images, f)
    
    return uploaded_images

def load_descriptions(descriptions_folder):
    """Load descriptions from text files in folder"""
    descriptions = {}
    
    # You may need to adjust this based on your description file format
    desc_files = [f for f in os.listdir(descriptions_folder) 
                 if f.lower().endswith(('.txt', '.json'))]
    
    for desc_file in desc_files:
        img_id = os.path.splitext(desc_file)[0]
        file_path = os.path.join(descriptions_folder, desc_file)
        
        # For txt files
        if desc_file.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                descriptions[img_id] = f.read().strip()
        
        # For json files (if your descriptions are in JSON)
        elif desc_file.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Adjust depending on your JSON structure
                descriptions[img_id] = data.get('description', '')
    
    return descriptions

def create_image_description_pairs(uploaded_images, descriptions):
    """Create pairs by joining images and descriptions by ID"""
    pairs = []
    
    for img_info in uploaded_images:
        img_id = img_info["image_id"]
        
        if img_id in descriptions:
            pairs.append({
                "image_id": img_id,
                "image_url": img_info["image_url"],
                "description": descriptions[img_id],
                "is_golden_set": False,  # Set to True for golden set items
                "metadata": {
                    "cloudinary_public_id": img_info["cloudinary_public_id"]
                },
                "created_at": datetime.datetime.now()
            })
        else:
            print(f"Warning: No description found for image ID {img_id}")
    
    return pairs

def import_pairs_to_mongodb(pairs):
    """Import the pairs to MongoDB collection"""
    if not pairs:
        print("No pairs to import!")
        return
    
    try:
        # Insert many pairs at once
        result = db.pairs.insert_many(pairs)
        print(f"Successfully imported {len(result.inserted_ids)} pairs to MongoDB")
        return result.inserted_ids
    except Exception as e:
        print(f"Error importing to MongoDB: {str(e)}")
        return None

def main():
    # 1. Upload images to Cloudinary
    print("Uploading images to Cloudinary...")
    uploaded_images = upload_images_to_cloudinary(IMAGES_FOLDER)
    
    # 2. Load descriptions
    print("Loading descriptions...")
    descriptions = load_descriptions(DESCRIPTIONS_FOLDER)
    
    # 3. Create image-description pairs
    print("Creating image-description pairs...")
    pairs = create_image_description_pairs(uploaded_images, descriptions)
    
    # 4. Import to MongoDB
    print("Importing pairs to MongoDB...")
    import_pairs_to_mongodb(pairs)
    
    print("Process completed!")

if __name__ == "__main__":
    main()