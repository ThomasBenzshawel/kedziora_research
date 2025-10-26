import os
import glob
import traceback
import sys
import logging
import subprocess
import time
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Set up logging (will be configured after parsing args)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Always log everything to the file
file_handler = logging.FileHandler("glb_conversion.log")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Console handler will be added after parsing arguments

# Process a single file using subprocess to isolate potential crashes
def convert_glb_to_ply(input_path, output_path, timeout=300):
    """
    Convert a single GLB file to PLY using a separate subprocess to isolate crashes
    
    Args:
        input_path: Path to the input GLB file
        output_path: Path to save the output PLY file
        timeout: Maximum time in seconds to wait for conversion (default: 5 minutes)
    
    Returns:
        True if conversion was successful, False otherwise
    """
    converter_script = "converter_script.py"
    
    logging.info(f"Starting conversion of {input_path}")
    
    try:
        # Run the conversion in a separate process with timeout
        process = subprocess.Popen(
            [sys.executable, converter_script, input_path, output_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Wait for process to complete with timeout
        try:
            stdout, stderr = process.communicate(timeout=timeout)
            
            # Log subprocess output
            for line in stdout.splitlines():
                logging.info(f"Subprocess: {line}")
            
            for line in stderr.splitlines():
                logging.error(f"Subprocess error: {line}")
            
            # Check return code
            if process.returncode == 0:
                logging.info(f"Successfully converted {input_path} to {output_path}")
                return True
            else:
                logging.error(f"Conversion failed with return code {process.returncode}")
                return False
                
        except subprocess.TimeoutExpired:
            # Kill the process if it times out
            process.kill()
            logging.error(f"Conversion timed out after {timeout} seconds")
            return False
            
    except Exception as e:
        logging.error(f"Error starting conversion process: {str(e)}")
        logging.error(traceback.format_exc())
        return False

# Worker function for multiprocessing
def worker_function(args):
    """
    Worker function to be used with ProcessPoolExecutor
    Args:
        args: Tuple containing (input_path, output_path, timeout)
    
    Returns:
        Tuple of (success, input_path, output_path)
    """
    input_path, output_path, timeout = args
    success = convert_glb_to_ply(input_path, output_path, timeout)
    return (success, input_path, output_path)

# Batch processing with progress tracking and parallelism
def batch_convert(input_dir, output_dir, extension='.ply', skip_existing=True, timeout=300, max_workers=None):
    os.makedirs(output_dir, exist_ok=True)
    
    # Get total number of files for progress tracking
    total_files = 0
    folders = []
    for folder in os.listdir(input_dir):
        folder_path = os.path.join(input_dir, folder)
        if os.path.isdir(folder_path):
            folders.append(folder)
            total_files += len(glob.glob(os.path.join(folder_path, '*.glb')))
    
    logging.info(f"Found {total_files} GLB files to process across {len(folders)} folders")
    
    # Create task list for parallel processing
    tasks = []
    skipped = 0
    
    for folder in sorted(folders):
        folder_path = os.path.join(input_dir, folder)
        
        # Create corresponding output folder
        output_folder = os.path.join(output_dir, folder)
        os.makedirs(output_folder, exist_ok=True)
        
        # Process all GLB files in the folder
        glb_files = glob.glob(os.path.join(folder_path, '*.glb'))
        for glb_file in sorted(glb_files):
            filename = os.path.basename(glb_file)
            name_without_ext = os.path.splitext(filename)[0]
            output_path = os.path.join(output_folder, name_without_ext + extension)
            
            # Skip if output already exists and skip_existing is True
            if skip_existing and os.path.exists(output_path):
                logging.info(f"Skipping {filename} - output already exists")
                skipped += 1
                continue
            
            # Add to task list
            tasks.append((glb_file, output_path, timeout))
    
    # Stats tracking
    to_process = len(tasks)
    total = to_process + skipped
    processed = 0
    success = 0
    failed = 0
    
    logging.info(f"Skipping {skipped} existing files")
    logging.info(f"Processing {to_process} files with {max_workers} workers")
    
    # Process files in parallel with progress bar
    with tqdm(total=to_process, desc="Converting GLB files") as pbar:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_path = {executor.submit(worker_function, task): task[0] for task in tasks}
            
            # Process results as they complete
            for future in as_completed(future_to_path):
                result, input_path, output_path = future.result()
                if result:
                    success += 1
                else:
                    failed += 1
                
                processed += 1
                pbar.update(1)
                
                # Log progress every 10 files
                if processed % 10 == 0:
                    logging.info(f"Progress: {processed}/{to_process} files (Success: {success}, Failed: {failed}, Skipped: {skipped})")

    # Final report
    logging.info("=" * 50)
    logging.info("Conversion complete!")
    logging.info(f"Total files: {total}")
    logging.info(f"Processed: {processed}")
    logging.info(f"Successful: {success}")
    logging.info(f"Failed: {failed}")
    logging.info(f"Skipped: {skipped}")
    logging.info("=" * 50)

if __name__ == "__main__":
    print("Converting GLB files to PLY...")
    
    # Parse command-line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Convert GLB files to PLY format')
    parser.add_argument('--input', type=str, default="/home/benzshawelt/.objaverse/hf-objaverse-v1/glbs/",
                        help='Input directory containing GLB files')
    parser.add_argument('--output', type=str, default="/home/benzshawelt/Kedziora/kedziora_research/layer_x_layer/data_gen/ply_files",
                        help='Output directory for PLY files')
    parser.add_argument('--skip-existing', action='store_true', help='Skip conversion if output file already exists')
    parser.add_argument('--timeout', type=int, default=300, help='Timeout in seconds for each conversion (default: 300)')
    parser.add_argument('--workers', type=int, default=None, help='Number of worker processes (default: CPU count)')
    parser.add_argument('--test', action='store_true', help='Run a single test conversion')
    parser.add_argument('-v', '--verbose', action='store_true', help='Show INFO level logs (default: only show errors)')
    args = parser.parse_args()
    
    # Set up console logging based on verbosity
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    if args.verbose:
        console_handler.setLevel(logging.INFO)
    else:
        console_handler.setLevel(logging.ERROR)
    logger.addHandler(console_handler)
    
    if args.test:
        # Run a single test conversion
        test_file = "/home/benzshawelt/.objaverse/hf-objaverse-v1/glbs/000-000/000a00944e294f7a94f95d420fdd45eb.glb"
        output_dir = args.output
        os.makedirs(output_dir, exist_ok=True)
        # Create the converter script first
        convert_glb_to_ply(test_file, os.path.join(output_dir, "test.ply"))
    else:
        # Run batch conversion with multi-threading
        batch_convert(args.input, args.output, skip_existing=args.skip_existing, 
                     timeout=args.timeout, max_workers=args.workers)