import pymeshlab

# Process a single file
def convert_glb_to_ply(input_path, output_path):
    ms = pymeshlab.MeshSet()
    
    # Load the GLB file
    ms.load_new_mesh(input_path)
    
    # Optional: Apply any necessary cleaning or processing
    # ms.apply_filter('meshing_remove_duplicate_vertices')
    
    # Save as PLY with vertex colors
    ms.save_current_mesh(        
            output_path,
            save_vertex_color=True,
            save_face_color=False,
            save_textures=False,  # Critical: Disable texture saving
            binary=True,  # Optional: Save as binary
    )
# # Example usage
# convert_glb_to_ply('model.glb', 'model.ply')

# For batch processing
import os
import glob

def batch_convert(input_dir, output_dir, extension='.ply'):
    os.makedirs(output_dir, exist_ok=True)
    for folder in os.listdir(input_dir):
        folder_path = os.path.join(input_dir, folder)
        if os.path.isdir(folder_path):
            for glb_file in glob.glob(os.path.join(folder_path, '*.glb')):
                filename = os.path.basename(glb_file)
                name_without_ext = os.path.splitext(filename)[0]
                output_path = os.path.join(output_dir, folder, name_without_ext + extension)
                convert_glb_to_ply(glb_file, output_path)
                print(f"Converted {filename} to {os.path.basename(output_path)}")


if __name__ == "__main__":
    print("Converting GLB files to PLY...")
    test = "/home/benzshawelt/.objaverse/hf-objaverse-v1/glbs/000-000/000a00944e294f7a94f95d420fdd45eb.glb"
    convert_glb_to_ply(test, "test.ply")
    # input_dir = "/home/benzshawelt/.objaverse/hf-objaverse-v1/glbs/"
    # output_dir = "/home/benzshawelt/kedziora_research/layer_x_layer/data_gen/ply_files"
    # batch_convert(input_dir, output_dir)