#!/usr/bin/env python3
"""
Coordinator script for managing parallel Objaverse rendering jobs
"""

import os
import json
import subprocess
import argparse
import time
from pathlib import Path

def count_objects(json_path):
    """Count total objects in the dataset"""
    with open(json_path, 'rt', encoding='utf-8') as f:
        object_paths = json.load(f)
    return len(object_paths)

def estimate_job_size(json_path, num_gpus):
    """Estimate how many objects each GPU will process"""
    total_objects = count_objects(json_path)
    objects_per_gpu = total_objects // num_gpus
    return total_objects, objects_per_gpu

def submit_jobs(num_gpus, sbatch_script="submit_render_job.sbatch"):
    """Submit the SLURM array job"""
    
    # Modify the sbatch script to use the correct array size
    with open(sbatch_script, 'r') as f:
        content = f.read()
    
    # Update array range
    new_content = content.replace(
        "#SBATCH --array=0-7", 
        f"#SBATCH --array=0-{num_gpus-1}"
    )
    
    # Write temporary sbatch script
    temp_script = f"temp_submit_{num_gpus}gpus.sbatch"
    with open(temp_script, 'w') as f:
        f.write(new_content)
    
    try:
        # Submit the job
        result = subprocess.run(['sbatch', temp_script], 
                               capture_output=True, text=True)
        if result.returncode == 0:
            job_id = result.stdout.strip().split()[-1]
            print(f"Job submitted successfully! Job ID: {job_id}")
            return job_id
        else:
            print(f"Error submitting job: {result.stderr}")
            return None
    finally:
        # Clean up temp script
        os.remove(temp_script)

def monitor_progress(output_dir, num_gpus):
    """Monitor job progress by checking status files"""
    while True:
        completed_workers = 0
        total_processed = 0
        total_errors = 0
        
        for i in range(num_gpus):
            status_file = os.path.join(output_dir, f"worker_{i}_status.txt")
            if os.path.exists(status_file):
                with open(status_file, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.startswith('processed:'):
                            total_processed += int(line.split(':')[1].strip())
                        elif line.startswith('errors:'):
                            total_errors += int(line.split(':')[1].strip())
                completed_workers += 1
        
        print(f"Progress: {completed_workers}/{num_gpus} workers completed")
        print(f"Total processed: {total_processed}, Total errors: {total_errors}")
        
        if completed_workers == num_gpus:
            print("All workers completed!")
            break
        
        time.sleep(60)  # Check every minute

def check_job_status(job_id):
    """Check SLURM job status"""
    try:
        result = subprocess.run(['squeue', '-j', job_id], 
                               capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:  # Header + at least one job
                return "RUNNING"
            else:
                return "COMPLETED"
        return "UNKNOWN"
    except:
        return "ERROR"

def gather_results(output_dir):
    """Gather final statistics from all workers"""
    total_processed = 0
    total_errors = 0
    worker_stats = {}
    
    # Collect stats from each worker
    for status_file in Path(output_dir).glob("worker_*_status.txt"):
        worker_id = status_file.stem.split('_')[1]
        with open(status_file, 'r') as f:
            stats = {}
            for line in f:
                key, value = line.strip().split(': ')
                stats[key] = int(value)
            worker_stats[worker_id] = stats
            total_processed += stats['processed']
            total_errors += stats['errors']
    
    # Write summary
    summary_file = os.path.join(output_dir, "job_summary.json")
    summary = {
        "total_processed": total_processed,
        "total_errors": total_errors,
        "worker_count": len(worker_stats),
        "worker_stats": worker_stats
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Final results:")
    print(f"  Total processed: {total_processed}")
    print(f"  Total errors: {total_errors}")
    print(f"  Success rate: {total_processed/(total_processed+total_errors)*100:.1f}%")
    print(f"  Summary saved to: {summary_file}")

def main():
    parser = argparse.ArgumentParser(description='Coordinate parallel Objaverse rendering')
    parser.add_argument('--num_gpus', type=int, required=False, 
                       help='Number of GPUs to use')
    parser.add_argument('--json_path', required=False, default="/home/ad.msoe.edu/benzshawelt/.objaverse/hf-objaverse-v1/object-paths.json",
                       help='Path to object-paths.json')
    parser.add_argument('--output_dir', default='../objaverse_images',
                       help='Output directory for images')
    parser.add_argument('--action', choices=['estimate', 'submit', 'monitor', 'summary'],
                       default='submit', help='Action to perform')
    parser.add_argument('--job_id', help='Job ID for monitoring')
    
    args = parser.parse_args()
    
    if args.action == 'estimate':
        total, per_gpu = estimate_job_size(args.json_path, args.num_gpus)
        print(f"Dataset analysis:")
        print(f"  Total objects: {total:,}")
        print(f"  Objects per GPU: {per_gpu:,}")
        print(f"  Last GPU will process: {total - (args.num_gpus-1)*per_gpu:,}")
        
    elif args.action == 'submit':
        # Create necessary directories
        os.makedirs('logs', exist_ok=True)
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Show estimate first
        total, per_gpu = estimate_job_size(args.json_path, args.num_gpus)
        print(f"Submitting job for {args.num_gpus} GPUs:")
        print(f"  Total objects: {total:,}")
        print(f"  Objects per GPU: ~{per_gpu:,}")
        
        job_id = submit_jobs(args.num_gpus)
        if job_id:
            print(f"\nTo monitor progress, run:")
            print(f"python3 {__file__} --action monitor --num_gpus {args.num_gpus} --output_dir {args.output_dir}")
            
    elif args.action == 'monitor':
        if args.job_id:
            status = check_job_status(args.job_id)
            print(f"Job {args.job_id} status: {status}")
        
        print("Monitoring worker progress...")
        monitor_progress(args.output_dir, args.num_gpus)
        
    elif args.action == 'summary':
        gather_results(args.output_dir)

if __name__ == "__main__":
    main()