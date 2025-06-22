import os
import argparse
import csv
import random
from pymongo import MongoClient
from datetime import datetime, timezone
from dotenv import load_dotenv
import pandas as pd

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Assign objects to users")
    parser.add_argument("--csv", required=True, help="CSV file containing userIds")
    parser.add_argument("--dry-run", action="store_true", help="Simulate assignments without updating MongoDB")
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
        db = client.objaverse  # Database name
        objects_collection = db.objects
        print(f"Connected to MongoDB database: objaverse")
    except Exception as e:
        print(f"Error connecting to MongoDB: {str(e)}")
        return
    
    # Read userIds from CSV
    userIds = []
    try:
        # Read the CSV file
        df = pd.read_csv(args.csv)
        # Check if the CSV has a 'userId' column
        if 'userId' not in df.columns:
            print("Error: CSV file must contain a 'userId' column")
            return
        # Extract userIds from the 'userId' column
        userIds = df['userId'].tolist()
        # Remove duplicates and strip whitespace
        userIds = list(set(userId.strip() for userId in userIds if isinstance(userId, str)))
        
        # We need exactly 7 users for this algorithm
        if len(userIds) != 7:
            print(f"Error: Expected 7 users, but found {len(userIds)} in the CSV")
            return
            
        print(f"Read {len(userIds)} userIds from CSV: {', '.join(userIds)}")
    except Exception as e:
        print(f"Error reading CSV file: {str(e)}")
        return
    
    # Get all object IDs from MongoDB
    try:
        all_objects = list(objects_collection.find({}, {"objectId": 1, "_id": 0}))
        all_object_ids = [obj["objectId"] for obj in all_objects]
        
        # Check if we have enough objects
        required_objects = 1085 + 135  # 155 unique per user × 7 users + 135 overlap
        if len(all_object_ids) < required_objects:
            print(f"Error: Not enough objects in database. Required: {required_objects}, Found: {len(all_object_ids)}")
            return
            
        print(f"Found {len(all_object_ids)} objects in database")
    except Exception as e:
        print(f"Error retrieving objects from MongoDB: {str(e)}")
        return
    
    # Randomly select objects for assignment
    random.shuffle(all_object_ids)
    unique_objects = all_object_ids[:1085]  # 155 unique per user × 7 users
    overlap_objects = all_object_ids[1085:1085+135]  # 135 objects for overlap
    
    # Create assignments for unique objects (155 per user)
    user_assignments = {}
    for i, userId in enumerate(userIds):
        start_idx = i * 155
        end_idx = start_idx + 155
        user_assignments[userId] = {
            "unique_objects": unique_objects[start_idx:end_idx],
            "overlap_objects": []  # Will be filled in the next step
        }
    
    # Create assignments for overlap objects
    # Track how many overlap objects each user has
    user_overlap_counts = {userId: 0 for userId in userIds}
    
    # Initially, all users are in the pool
    user_pool = userIds.copy()
    
    # Note: There are two constraints that we're trying to balance:
    # 1. Each overlap object should be assigned to 3 users
    # 2. Each user should get at most 45 overlap objects
    #
    # Our approach: Process all 135 objects, and for each one:
    # - If there are at least 3 users in the pool, assign it to 3 random users
    # - If there are fewer than 3 users in the pool, assign it to all remaining users
    # - Remove users from the pool once they reach 45 overlap objects
    
    # Distribute overlap objects
    for obj_idx, obj_id in enumerate(overlap_objects):
        # If no users left in pool, we can't assign any more objects
        if not user_pool:
            print(f"Warning: No users left in pool to assign objects. Stopping at object {obj_idx} of 135")
            break
        
        # Get number of users to assign this object to (up to 3, or all remaining users if fewer)
        num_users_to_assign = min(3, len(user_pool))
        
        # Select random users from the pool
        selected_users = random.sample(user_pool, num_users_to_assign)
        
        # Assign the object to these users and update counts
        for userId in selected_users:
            user_assignments[userId]["overlap_objects"].append(obj_id)
            user_overlap_counts[userId] += 1
            
            # If user now has 45 overlap objects, remove them from the pool
            if user_overlap_counts[userId] >= 45:
                user_pool.remove(userId)
                print(f"User {userId} has reached 45 overlap objects and is removed from the pool ({len(user_pool)} users remaining)")
        
        # Progress update every 20 objects
        if (obj_idx + 1) % 20 == 0:
            print(f"Processed {obj_idx + 1} of 135 overlap objects. Users in pool: {len(user_pool)}")
    
    # Print assignment summary
    print("\nAssignment Summary:")
    for userId, assignments in user_assignments.items():
        unique_count = len(assignments["unique_objects"])
        overlap_count = len(assignments["overlap_objects"])
        total_count = unique_count + overlap_count
        print(f"User {userId}: {unique_count} unique + {overlap_count} overlap = {total_count} total objects")
    
    # Verify all overlap objects were assigned to at least one user
    overlap_assignment_counts = {}
    for userId, assignments in user_assignments.items():
        for obj_id in assignments["overlap_objects"]:
            if obj_id in overlap_assignment_counts:
                overlap_assignment_counts[obj_id] += 1
            else:
                overlap_assignment_counts[obj_id] = 1
    
    assigned_overlap_objects = len(overlap_assignment_counts)
    print(f"\nObjects with overlap assignments: {assigned_overlap_objects} of 135")
    
    # Count objects by number of assignments
    assignment_distribution = {}
    for count in overlap_assignment_counts.values():
        if count in assignment_distribution:
            assignment_distribution[count] += 1
        else:
            assignment_distribution[count] = 1
    
    print("Distribution of overlap objects by number of assignments:")
    for count, num_objects in sorted(assignment_distribution.items()):
        print(f"  {num_objects} objects assigned to {count} user{'s' if count != 1 else ''}")
    
    # If this is a dry run, don't update MongoDB
    if args.dry_run:
        print("\nThis was a dry run. No changes were made to MongoDB.")

        # export to csv for manually checking
        output_file = "user_assignments.csv"
        with open(output_file, 'w', newline='') as csvfile:
            fieldnames = ['userId', 'unique_objects', 'overlap_objects']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            # go by object id and have a list of users for each object
            for userId, assignments in user_assignments.items():
                writer.writerow({
                    'userId': userId,
                    'unique_objects': ', '.join(assignments["unique_objects"]),
                    'overlap_objects': ', '.join(assignments["overlap_objects"])
                })
        print(f"User assignments exported to {output_file}")
        print("Exiting without making any changes to MongoDB.")
        return
    
    # Now, update the MongoDB with these assignments
    now = datetime.now(timezone.utc)
    
    # Define a function to update assignments for a user
    def update_object_assignments(object_id, userId):
        objects_collection.update_one(
            {"objectId": object_id},
            {
                "$push": {
                    "assignments": {
                        "userId": userId,
                        "assignedAt": now
                    }
                },
                "$set": {
                    "updatedAt": now
                }
            }
        )
    
    # Update MongoDB with assignments
    assignment_count = 0
    try:
        for userId, assignments in user_assignments.items():
            # Combine unique and overlap objects for this user
            all_user_objects = assignments["unique_objects"] + assignments["overlap_objects"]
            
            for obj_id in all_user_objects:
                update_object_assignments(obj_id, userId)
                assignment_count += 1
                
            print(f"Updated {len(all_user_objects)} assignments for user {userId}")
        
        print(f"\nSuccessfully created {assignment_count} total object assignments")
    except Exception as e:
        print(f"Error updating MongoDB with assignments: {str(e)}")
    
if __name__ == "__main__":
    main()