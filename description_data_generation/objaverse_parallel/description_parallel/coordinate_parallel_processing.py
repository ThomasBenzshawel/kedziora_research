#!/usr/bin/env python3
"""
Coordinator script for parallel LLaVA processing
Handles job submission, monitoring, and result aggregation
"""

import subprocess
import time
import os
import json
import glob
import argparse
import sys
from datetime import datetime
import re

class ParallelProcessingCoordinator:
    def __init__(self, total_workers=8, target_folder="../objaverse_images"):
        self.total_workers = total_workers
        self.target_folder = target_folder
        self.job_id = None
        self.slurm_script = "run_llava_parallel.slurm"
        self.checkpoint_dir = "checkpoints"
        self.logs_dir = "logs"
        
        # Create necessary directories
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs("objaverse_descriptions", exist_ok=True)

    def count_total_folders(self):
        """Count total folders that need processing"""
        total_folders = 0
        processed_folders = 0
        
        if not os.path.exists(self.target_folder):
            print(f"Error: Target folder {self.target_folder} not found!")
            return 0, 0
        
        for folder in os.listdir(self.target_folder):
            folder_path = os.path.join(self.target_folder, folder)
            if not os.path.isdir(folder_path):
                continue
                
            try:
                num_images = len(os.listdir(folder_path))
                if num_images != 6:
                    continue
                    
                total_folders += 1
                
                # Check if already processed
                output_file = f"./objaverse_descriptions/{folder}.txt"
                if os.path.exists(output_file):
                    processed_folders += 1
                    
            except Exception:
                continue
        
        return total_folders, processed_folders

    def submit_job(self):
        """Submit SLURM job array"""
        print("Submitting SLURM job array...")
        
        # Update SLURM script with current configuration
        self.update_slurm_script()
        
        cmd = ["sbatch", self.slurm_script]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            output = result.stdout.strip()
            
            # Extract job ID from sbatch output
            match = re.search(r'Submitted batch job (\d+)', output)
            if match:
                self.job_id = match.group(1)
                print(f"Job submitted successfully! Job ID: {self.job_id}")
                return True
            else:
                print(f"Error: Could not extract job ID from output: {output}")
                return False
                
        except subprocess.CalledProcessError as e:
            print(f"Error submitting job: {e}")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
            return False

    def update_slurm_script(self):
        """Update SLURM script with current configuration"""
        if not os.path.exists(self.slurm_script):
            print(f"Error: SLURM script {self.slurm_script} not found!")
            return
        
        with open(self.slurm_script, 'r') as f:
            content = f.read()
        
        # Update array size and concurrent limit
        array_line = f"#SBATCH --array=0-{self.total_workers-1}%{self.total_workers}"
        content = re.sub(r'#SBATCH --array=\d+-\d+%\d+', array_line, content)
        
        # Update total workers
        content = re.sub(r'TOTAL_WORKERS=\d+', f'TOTAL_WORKERS={self.total_workers}', content)
        
        # Update target folder
        content = re.sub(r'TARGET_FOLDER="[^"]*"', f'TARGET_FOLDER="{self.target_folder}"', content)
        
        with open(self.slurm_script, 'w') as f:
            f.write(content)

    def monitor_job(self, check_interval=60):
        """Monitor job progress"""
        if not self.job_id:
            print("No job ID available for monitoring")
            return
        
        print(f"Monitoring job {self.job_id}...")
        print("Press Ctrl+C to stop monitoring (job will continue running)")
        
        try:
            start_time = time.time()
            last_status_check = 0
            
            while True:
                current_time = time.time()
                
                # Check SLURM job status
                if current_time - last_status_check >= check_interval:
                    job_status = self.get_job_status()
                    worker_status = self.get_worker_status()
                    processing_stats = self.get_processing_stats()
                    
                    elapsed = current_time - start_time
                    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Status Update (Elapsed: {elapsed/3600:.1f}h)")
                    print(f"Job Status: {job_status}")
                    print(f"Worker Status: {worker_status}")
                    print(f"Processing Stats: {processing_stats}")
                    
                    # Check if job is complete
                    if job_status in ["COMPLETED", "FAILED", "CANCELLED"]:
                        print(f"Job finished with status: {job_status}")
                        break
                        
                    last_status_check = current_time
                
                time.sleep(10)  # Check every 10 seconds for quick status updates
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user. Job continues running.")
            print(f"To check status later, run: squeue -j {self.job_id}")

    def get_job_status(self):
        """Get SLURM job status"""
        try:
            cmd = ["squeue", "-j", self.job_id, "--noheader", "--format=%T"]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            status = result.stdout.strip()
            return status if status else "UNKNOWN"
        except subprocess.CalledProcessError:
            return "NOT_FOUND"

    def get_worker_status(self):
        """Get status of individual workers"""
        completed = 0
        failed = 0
        running = 0
        
        for worker_id in range(self.total_workers):
            checkpoint_file = f"{self.checkpoint_dir}/worker_{worker_id}_checkpoint.txt"
            if os.path.exists(checkpoint_file):
                with open(checkpoint_file, 'r') as f:
                    content = f.read()
                    if "completed successfully" in content:
                        completed += 1
                    elif "failed" in content:
                        failed += 1
                    elif "starting" in content:
                        running += 1
        
        return f"{completed} completed, {running} running, {failed} failed of {self.total_workers} total"

    def get_processing_stats(self):
        """Get processing statistics"""
        total_folders, initially_processed = self.count_total_folders()
        
        # Count current processed folders
        current_processed = 0
        if os.path.exists("objaverse_descriptions"):
            current_processed = len([f for f in os.listdir("objaverse_descriptions") 
                                   if f.endswith('.txt')])
        
        newly_processed = current_processed - initially_processed
        remaining = total_folders - current_processed
        
        progress_percent = (current_processed / total_folders) * 100 if total_folders > 0 else 0
        
        return (f"{current_processed}/{total_folders} processed ({progress_percent:.1f}%), "
                f"{newly_processed} newly processed, {remaining} remaining")

    def cleanup_checkpoints(self):
        """Clean up old checkpoint files"""
        print("Cleaning up checkpoint files...")
        checkpoint_files = glob.glob(f"{self.checkpoint_dir}/worker_*_checkpoint.txt")
        for file in checkpoint_files:
            os.remove(file)
        print(f"Removed {len(checkpoint_files)} checkpoint files")

    def merge_metadata_caches(self):
        """Merge all worker metadata caches into main cache"""
        print("Merging metadata caches...")
        
        main_cache_file = "./objaverse_metadata_cache.json"
        worker_cache_pattern = "./objaverse_metadata_cache_worker_*.json"
        
        # Load main cache
        main_cache = {}
        main_search_times = {}
        
        if os.path.exists(main_cache_file):
            with open(main_cache_file, 'r') as f:
                try:
                    data = json.load(f)
                    main_cache = data.get('cache', {})
                    main_search_times = data.get('search_times', {})
                except:
                    pass
        
        # Merge worker caches
        merged_count = 0
        worker_files = glob.glob(worker_cache_pattern)
        
        for worker_cache_file in worker_files:
            try:
                with open(worker_cache_file, 'r') as f:
                    data = json.load(f)
                    worker_cache = data.get('cache', {})
                    worker_search_times = data.get('search_times', {})
                    
                    for uid, name in worker_cache.items():
                        if uid not in main_cache:
                            main_cache[uid] = name
                            main_search_times[uid] = worker_search_times.get(uid, 0)
                            merged_count += 1
                            
            except Exception as e:
                print(f"Error processing {worker_cache_file}: {e}")
        
        # Save merged cache
        if merged_count > 0:
            merged_data = {
                'cache': main_cache,
                'search_times': main_search_times,
                'last_updated': time.time(),
                'merged_workers': len(worker_files),
                'total_entries': len(main_cache)
            }
            
            with open(main_cache_file, 'w') as f:
                json.dump(merged_data, f, indent=2)
            
            print(f"Merged {merged_count} new metadata entries from {len(worker_files)} worker caches")
            print(f"Total metadata entries: {len(main_cache)}")
        else:
            print("No new metadata entries to merge")
        
        # Clean up worker cache files
        for worker_file in worker_files:
            os.remove(worker_file)
        print(f"Cleaned up {len(worker_files)} worker cache files")

    def generate_summary_report(self):
        """Generate summary report of processing results"""
        print("\n" + "="*60)
        print("PARALLEL PROCESSING SUMMARY REPORT")
        print("="*60)
        
        total_folders, initially_processed = self.count_total_folders()
        current_processed = 0
        
        if os.path.exists("objaverse_descriptions"):
            current_processed = len([f for f in os.listdir("objaverse_descriptions") 
                                   if f.endswith('.txt')])
        
        newly_processed = current_processed - initially_processed
        success_rate = (newly_processed / (total_folders - initially_processed)) * 100 if total_folders > initially_processed else 0
        
        print(f"Total folders found: {total_folders}")
        print(f"Initially processed: {initially_processed}")
        print(f"Newly processed: {newly_processed}")
        print(f"Success rate: {success_rate:.1f}%")
        print(f"Final processed: {current_processed}")
        print(f"Remaining: {total_folders - current_processed}")
        
        # Worker status summary
        print("\nWorker Status Summary:")
        for worker_id in range(self.total_workers):
            checkpoint_file = f"{self.checkpoint_dir}/worker_{worker_id}_checkpoint.txt"
            if os.path.exists(checkpoint_file):
                with open(checkpoint_file, 'r') as f:
                    content = f.read()
                    if "completed successfully" in content:
                        status = "✓ COMPLETED"
                    elif "failed" in content:
                        status = "✗ FAILED"
                    elif "starting" in content:
                        status = "⟳ RUNNING"
                    else:
                        status = "? UNKNOWN"
            else:
                status = "- NOT STARTED"
            print(f"  Worker {worker_id}: {status}")
        
        # Check for log files with errors
        error_logs = glob.glob(f"{self.logs_dir}/*_parallel_*.err")
        if error_logs:
            print(f"\nWarning: Found {len(error_logs)} error log files")
            print("Check these files for any issues:")
            for log_file in error_logs[:5]:  # Show first 5
                print(f"  {log_file}")
        
        print("="*60)

def main():
    parser = argparse.ArgumentParser(description='Coordinate parallel LLaVA processing')
    parser.add_argument('--workers', type=int, default=8, help='Number of parallel workers')
    parser.add_argument('--target_folder', type=str, default='../objaverse_images',
                       help='Target folder with images')
    parser.add_argument('--action', choices=['submit', 'monitor', 'status', 'cleanup', 'full'],
                       default='full', help='Action to perform')
    parser.add_argument('--job_id', type=str, help='Existing job ID to monitor')
    parser.add_argument('--check_interval', type=int, default=60,
                       help='Monitoring check interval in seconds')
    
    args = parser.parse_args()
    
    coordinator = ParallelProcessingCoordinator(args.workers, args.target_folder)
    
    if args.job_id:
        coordinator.job_id = args.job_id
    
    if args.action == 'submit':
        success = coordinator.submit_job()
        if success:
            print(f"Job submitted. Monitor with: {sys.argv[0]} --action monitor --job_id {coordinator.job_id}")
    
    elif args.action == 'monitor':
        if not coordinator.job_id:
            print("No job ID provided or available")
            sys.exit(1)
        coordinator.monitor_job(args.check_interval)
    
    elif args.action == 'status':
        coordinator.generate_summary_report()
    
    elif args.action == 'cleanup':
        coordinator.cleanup_checkpoints()
        coordinator.merge_metadata_caches()
        print("Cleanup completed")
    
    elif args.action == 'full':
        # Full workflow: submit, monitor, cleanup
        print("Starting full parallel processing workflow...")
        
        # Show initial status
        total_folders, initially_processed = coordinator.count_total_folders()
        print(f"Initial status: {initially_processed}/{total_folders} folders already processed")
        
        if initially_processed >= total_folders:
            print("All folders already processed!")
            sys.exit(0)
        
        # Submit job
        success = coordinator.submit_job()
        if not success:
            sys.exit(1)
        
        # Monitor job
        try:
            coordinator.monitor_job(args.check_interval)
        except KeyboardInterrupt:
            print("Monitoring interrupted. Job continues running.")
        
        # Final cleanup and report
        coordinator.merge_metadata_caches()
        coordinator.generate_summary_report()
        coordinator.cleanup_checkpoints()

if __name__ == "__main__":
    main()