import os
import argparse
import glob
from pymongo import MongoClient
from datetime import datetime, timezone
import uuid
from dotenv import load_dotenv

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Upload object descriptions to MongoDB")
    parser.add_argument("--dir", required=True, help="Directory containing object description files")
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Get MongoDB connection details
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        print("Error: MONGO_URI not found in .env file")
        return
    
    # Connect to MongoDB
    try:
        client = MongoClient(mongo_uri)
        db = client.objaverse  # Database name from the original code
        objects_collection = db.objects
        print(f"Connected to MongoDB database: objaverse")
    except Exception as e:
        print(f"Error connecting to MongoDB: {str(e)}")
        return
    
    # Process description files
    description_files = glob.glob(os.path.join(args.dir, "*.txt"))
    
    if not description_files:
        print(f"No .txt files found in directory: {args.dir}")
        return
    
    print(f"Found {len(description_files)} description files to process")
    
    # Counters for statistics
    created_count = 0
    updated_count = 0
    error_count = 0
    
    for file_path in description_files:
        try:
            # Extract object ID from filename
            file_name = os.path.basename(file_path)
            object_id = os.path.splitext(file_name)[0]
            
            # Read description from file
            with open(file_path, 'r', encoding='utf-8') as file:
                description = file.read().strip()
            
            if not description:
                print(f"Warning: Empty description for object {object_id}")
                continue
            
            # Check if object already exists
            existing_object = objects_collection.find_one({"objectId": object_id})
            now = datetime.now(timezone.utc)
            
            if existing_object:
                # Update existing object
                result = objects_collection.update_one(
                    {"objectId": object_id},
                    {
                        "$set": {
                            "description": description,
                            "updatedAt": now
                        }
                    }
                )
                
                if result.modified_count > 0:
                    print(f"Updated description for object: {object_id}")
                    updated_count += 1
                else:
                    print(f"No changes made to object: {object_id}")
            else:
                # Create new object with minimal required fields
                new_object = {
                    "objectId": object_id,
                    "description": description,
                    "category": "Uncategorized",  # Default category
                    "images": [],
                    "ratings": [],
                    "assignments": [],
                    "createdAt": now,
                    "updatedAt": now
                }
                
                result = objects_collection.insert_one(new_object)
                print(f"Created new object: {object_id}")
                created_count += 1
                
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
            error_count += 1
    
    # Print summary
    print("\nUpload Summary:")
    print(f"Total files processed: {len(description_files)}")
    print(f"Objects created: {created_count}")
    print(f"Objects updated: {updated_count}")
    print(f"Errors: {error_count}")
    
if __name__ == "__main__":
    main()