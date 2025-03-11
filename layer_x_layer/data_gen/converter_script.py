import sys
import pymeshlab
import gc
import os

def convert_single_file(input_path, output_path):
    try:
        ms = pymeshlab.MeshSet()
        print(f"Loading {input_path}...")
        ms.load_new_mesh(input_path, load_in_a_single_layer=False)
        
        # Get the number of meshes in the set
        mesh_count = ms.mesh_number()
        print(f"Successfully loaded {input_path} with {mesh_count} meshes/parts")
        
        # Check if we have multiple meshes
        if mesh_count > 1:
            # Try to save a merged version
            try:
                # Flatten all meshes into one if there are multiple parts
                    print(f"Flattening {mesh_count} meshes into a single file...")
                    ms.generate_by_merging_visible_meshes()
                    
                    # Save the flattened mesh as the main output
                    ms.save_current_mesh(
                        output_path,
                        save_vertex_color=True,
                        save_face_color=False,
                        
                        save_textures=False,
                        binary=True,
                    )
                    print(f"Successfully saved flattened mesh to {output_path}")
            except Exception as e:
                # If we can't flatten the meshes, save them individually
                print(f"Warning: Could not create flattened version: {str(e)}")

                base_name, ext = os.path.splitext(output_path)
                
                for i in range(mesh_count):
                    # Set the current mesh to work with
                    ms.set_current_mesh(i)
                    
                    # Create part-specific filename
                    part_output = f"{base_name}_part{i}{ext}"
                    
                    print(f"Processing part {i+1}/{mesh_count} - Vertices: {ms.current_mesh().vertex_number()}, Faces: {ms.current_mesh().face_number()}")
                    
                    # Save current part
                    ms.save_current_mesh(
                        part_output,
                        save_vertex_color=True,
                        save_face_color=False,
                        save_textures=False,
                        binary=True,
                    )
                    print(f"Successfully saved part {i+1} to {part_output}")
                # Don't consider this a failure since we saved the individual parts
                
        else:
            # For single-part objects, save directly
            print(f"Vertices: {ms.current_mesh().vertex_number()}, Faces: {ms.current_mesh().face_number()}")
            
            # Save as PLY with vertex colors
            ms.save_current_mesh(
                output_path,
                save_vertex_color=True,
                save_face_color=False,
                save_textures=False,
                binary=True,
            )
            print(f"Successfully saved {output_path}")
        
        # Clean up resources
        del ms
        gc.collect()
        return 0
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python converter_script.py input_file output_file")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    sys.exit(convert_single_file(input_file, output_file))