import random
import multiprocessing as mp
import json
import os
import subprocess
import logging
import time
import tempfile
from typing import Any, Dict, Hashable, List, Optional
import objaverse
import pandas as pd
import objaverse.xl as oxl
from pathlib import Path
from functools import partial
import traceback
import dotenv

# Set up logging
def setup_logging(job_id: int = 1) -> logging.Logger:
    """Set up logging with job-specific log files."""
    log_filename = f'objaverse_download_job_{job_id}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# Configuration
dotenv.load_dotenv()  # Load environment variables from .env file if present
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
os.environ["GITHUB_TOKEN"] = GITHUB_TOKEN
os.environ["Github_token"] = GITHUB_TOKEN

def setup_custom_temp_dir(base_dir: str = None, job_id: int = 1):
    """Set up custom temporary directory with more space."""
    if base_dir is None:
        # Use home directory or current working directory with job ID
        base_dir = os.path.expanduser(f"~/temp/job_{job_id}") if os.path.exists(os.path.expanduser("~")) else f"./temp/job_{job_id}"
    
    # Create temp directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)
    
    # Set environment variables for temporary directory
    os.environ['TMPDIR'] = base_dir
    os.environ['TMP'] = base_dir
    os.environ['TEMP'] = base_dir
    tempfile.tempdir = base_dir
    
    logger.info(f"Set temporary directory to: {base_dir}")
    
    # Check available space
    statvfs = os.statvfs(base_dir)
    available_gb = (statvfs.f_frsize * statvfs.f_bavail) / (1024**3)
    logger.info(f"Available space in temp directory: {available_gb:.2f} GB")
    
    if available_gb < 10:
        logger.warning(f"Low disk space warning: Only {available_gb:.2f} GB available")
    
    return base_dir

def cleanup_temp_files(temp_dir: str, aggressive: bool = False):
    """
    Clean up temporary files periodically.
    
    Args:
        temp_dir: Directory to clean
        aggressive: If True, removes ALL files in temp_dir. If False, only removes tmp files.
    """
    try:
        import shutil
        
        if not os.path.exists(temp_dir):
            if logger:
                logger.info(f"Temp directory does not exist: {temp_dir}")
            return
        
        if aggressive:
            # Remove everything in temp_dir (but keep the directory itself)
            if logger:
                logger.info(f"Performing aggressive cleanup of {temp_dir}")
            
            removed_count = 0
            for item in os.listdir(temp_dir):
                item_path = os.path.join(temp_dir, item)
                try:
                    if os.path.isfile(item_path) or os.path.islink(item_path):
                        os.unlink(item_path)
                        removed_count += 1
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                        removed_count += 1
                except Exception as e:
                    if logger:
                        logger.warning(f"Failed to delete {item_path}: {e}")
            
            if logger:
                logger.info(f"Aggressive cleanup completed: removed {removed_count} items from {temp_dir}")
        else:
            # Only remove files that look like temp files
            temp_files = []
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    if file.startswith('tmp') or file.endswith('.tmp'):
                        temp_files.append(os.path.join(root, file))
            
            for temp_file in temp_files:
                try:
                    os.remove(temp_file)
                except:
                    pass
            
            if logger:
                logger.info(f"Cleaned up {len(temp_files)} temporary files")
    except Exception as e:
        if logger:
            logger.warning(f"Error cleaning temp files: {e}")

def manual_cleanup_all_temp_dirs(base_temp_dir: str = None):
    """
    Manually clean ALL temp directories from all jobs.
    Useful for cleaning up after crashes or before starting fresh.
    
    Args:
        base_temp_dir: Base directory containing temp folders. 
                       Default: ~/objaverse_temp
    
    Usage:
        python objaverse_download_fixed.py --manual-cleanup
    """
    import shutil
    
    if base_temp_dir is None:
        base_temp_dir = os.path.expanduser("~/objaverse_temp")
    
    if not os.path.exists(base_temp_dir):
        print(f"No temp directory found at: {base_temp_dir}")
        return
    
    print(f"\n{'='*60}")
    print(f"MANUAL CLEANUP: Removing all temp directories")
    print(f"Base directory: {base_temp_dir}")
    print(f"{'='*60}\n")
    
    removed_count = 0
    total_size = 0
    
    # Calculate size first
    for item in os.listdir(base_temp_dir):
        item_path = os.path.join(base_temp_dir, item)
        if os.path.isdir(item_path):
            # Calculate directory size
            for dirpath, dirnames, filenames in os.walk(item_path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    try:
                        total_size += os.path.getsize(filepath)
                    except:
                        pass
    
    total_size_gb = total_size / (1024**3)
    print(f"Found {total_size_gb:.2f} GB of temp data")
    
    # Remove directories
    for item in os.listdir(base_temp_dir):
        item_path = os.path.join(base_temp_dir, item)
        if os.path.isdir(item_path):
            try:
                print(f"  Removing: {item}")
                shutil.rmtree(item_path)
                removed_count += 1
            except Exception as e:
                print(f"  Failed to remove {item}: {e}")
    
    print(f"\n{'='*60}")
    print(f"Cleanup complete!")
    print(f"  Removed {removed_count} directories")
    print(f"  Freed ~{total_size_gb:.2f} GB")
    print(f"{'='*60}\n")

def setup_git_lfs_with_token():
    """Configure git LFS to use GitHub token for authentication."""
    try:
        # Set up git credentials for GitHub
        subprocess.run([
            "git", "config", "--global", "credential.helper", "store"
        ], check=True)
        
        # Create credentials file with token
        home_dir = Path.home()
        git_credentials_path = home_dir / ".git-credentials"
        
        # Add GitHub credentials
        with open(git_credentials_path, "a") as f:
            f.write(f"https://{GITHUB_TOKEN}:x-oauth-basic@github.com\n")
        
        # Configure git LFS
        subprocess.run([
            "git", "config", "--global", "lfs.transfer.maxretries", "3"
        ], check=True)
        
        subprocess.run([
            "git", "config", "--global", "lfs.transfer.maxretrydelay", "10"
        ], check=True)
        
        logger.info("Git LFS configured with GitHub token")
        
    except subprocess.CalledProcessError as e:
        logger.warning(f"Failed to configure git LFS: {e}")
    except Exception as e:
        logger.warning(f"Error setting up git credentials: {e}")

class CheckpointManager:
    """Manage checkpoints for resumable downloads."""
    def __init__(self, job_id: int, checkpoint_dir: str = "./checkpoints"):
        self.job_id = job_id
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint_job_{job_id}.json")
        self.completed_objects = set()
        self.failed_objects = set()
        self.load_checkpoint()
    
    def load_checkpoint(self):
        """Load existing checkpoint if available."""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r') as f:
                    data = json.load(f)
                    self.completed_objects = set(data.get('completed_objects', []))
                    self.failed_objects = set(data.get('failed_objects', []))
                    logger.info(f"Job {self.job_id}: Loaded checkpoint with {len(self.completed_objects)} completed and {len(self.failed_objects)} failed objects")
            except Exception as e:
                logger.warning(f"Job {self.job_id}: Failed to load checkpoint: {e}")
    
    def save_checkpoint(self):
        """Save current checkpoint."""
        try:
            data = {
                'job_id': self.job_id,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'completed_objects': list(self.completed_objects),
                'failed_objects': list(self.failed_objects),
                'total_completed': len(self.completed_objects),
                'total_failed': len(self.failed_objects)
            }
            with open(self.checkpoint_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Job {self.job_id}: Failed to save checkpoint: {e}")
    
    def mark_completed(self, file_identifier: str):
        """Mark an object as completed."""
        self.completed_objects.add(str(file_identifier))
    
    def mark_failed(self, file_identifier: str):
        """Mark an object as failed."""
        self.failed_objects.add(str(file_identifier))
    
    def is_completed(self, file_identifier: str) -> bool:
        """Check if object is already completed."""
        return str(file_identifier) in self.completed_objects
    
    def should_skip(self, file_identifier: str, download_dir: str) -> bool:
        """Check if object should be skipped (already downloaded or exists on disk)."""
        file_id = str(file_identifier)
        
        # Check checkpoint first
        if file_id in self.completed_objects:
            return True
        
        # Check if file exists on disk (in case checkpoint was lost)
        # This is a basic check - adjust path logic based on your download structure
        possible_paths = [
            os.path.join(download_dir, file_id),
            os.path.join(download_dir, f"{file_id}.glb"),
            os.path.join(download_dir, f"{file_id}.obj"),
        ]
        for path in possible_paths:
            if os.path.exists(path):
                logger.debug(f"Job {self.job_id}: Found existing file for {file_id}, marking as completed")
                self.mark_completed(file_id)
                return True
        
        return False

class ProgressTracker:
    """Track and save download progress periodically."""
    def __init__(self, job_id: int, total_objects: int, output_dir: str = "./output"):
        self.job_id = job_id
        self.total_objects = total_objects
        self.output_dir = output_dir
        self.progress_file = os.path.join(output_dir, f"job_{job_id}", f"progress_job_{job_id}.json")
        os.makedirs(os.path.dirname(self.progress_file), exist_ok=True)
        
        self.start_time = time.time()
        self.objects_processed = 0
        self.objects_successful = 0
        self.objects_failed = 0
        self.current_batch = 0
        self.total_batches = 0
        self.last_save_time = time.time()
    
    def update(self, batch_num: int = None, successful: int = 0, failed: int = 0):
        """Update progress counters."""
        if batch_num is not None:
            self.current_batch = batch_num
        self.objects_successful += successful
        self.objects_failed += failed
        self.objects_processed = self.objects_successful + self.objects_failed
    
    def save_progress(self, force: bool = False):
        """Save progress to file (with throttling unless forced)."""
        current_time = time.time()
        
        # Only save every 30 seconds unless forced
        if not force and (current_time - self.last_save_time) < 30:
            return
        
        elapsed_time = current_time - self.start_time
        objects_per_sec = self.objects_processed / elapsed_time if elapsed_time > 0 else 0
        remaining_objects = self.total_objects - self.objects_processed
        estimated_remaining_sec = remaining_objects / objects_per_sec if objects_per_sec > 0 else 0
        
        progress_data = {
            'job_id': self.job_id,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'progress': {
                'current_batch': self.current_batch,
                'total_batches': self.total_batches,
                'objects_processed': self.objects_processed,
                'objects_successful': self.objects_successful,
                'objects_failed': self.objects_failed,
                'total_objects': self.total_objects,
                'percent_complete': (self.objects_processed / self.total_objects * 100) if self.total_objects > 0 else 0
            },
            'timing': {
                'elapsed_seconds': round(elapsed_time, 2),
                'elapsed_formatted': self._format_duration(elapsed_time),
                'objects_per_second': round(objects_per_sec, 2),
                'estimated_remaining_seconds': round(estimated_remaining_sec, 2),
                'estimated_remaining_formatted': self._format_duration(estimated_remaining_sec),
                'estimated_completion_time': time.strftime('%Y-%m-%d %H:%M:%S', 
                    time.localtime(current_time + estimated_remaining_sec))
            },
            'rates': {
                'success_rate': (self.objects_successful / self.objects_processed * 100) if self.objects_processed > 0 else 0,
                'failure_rate': (self.objects_failed / self.objects_processed * 100) if self.objects_processed > 0 else 0
            }
        }
        
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2)
            self.last_save_time = current_time
        except Exception as e:
            logger.warning(f"Job {self.job_id}: Failed to save progress: {e}")
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format."""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            return f"{int(seconds/60)}m {int(seconds%60)}s"
        else:
            hours = int(seconds / 3600)
            minutes = int((seconds % 3600) / 60)
            return f"{hours}h {minutes}m"
    
    def log_progress(self):
        """Log current progress to console."""
        elapsed_time = time.time() - self.start_time
        objects_per_sec = self.objects_processed / elapsed_time if elapsed_time > 0 else 0
        percent_complete = (self.objects_processed / self.total_objects * 100) if self.total_objects > 0 else 0
        
        logger.info(f"Job {self.job_id}: Progress - {self.current_batch}/{self.total_batches} batches | "
                   f"{self.objects_processed}/{self.total_objects} objects ({percent_complete:.1f}%) | "
                   f"{objects_per_sec:.2f} obj/s | "
                   f"Success: {self.objects_successful}, Failed: {self.objects_failed}")

def check_disk_space(path: str, min_gb: float = 10.0, job_id: int = 1) -> dict:
    """
    Check available disk space and return status.
    
    Args:
        path: Path to check disk space for
        min_gb: Minimum GB required (warning threshold)
        job_id: Job ID for logging
    
    Returns:
        dict with 'available_gb', 'warning', 'critical' status
    """
    try:
        statvfs = os.statvfs(path)
        available_gb = (statvfs.f_frsize * statvfs.f_bavail) / (1024**3)
        
        status = {
            'available_gb': round(available_gb, 2),
            'warning': available_gb < min_gb,
            'critical': available_gb < (min_gb / 2),  # Critical at half the warning threshold
            'path': path
        }
        
        if status['critical']:
            logger.error(f"Job {job_id}: CRITICAL - Only {available_gb:.2f} GB available at {path}")
        elif status['warning']:
            logger.warning(f"Job {job_id}: WARNING - Only {available_gb:.2f} GB available at {path}")
        
        return status
    except Exception as e:
        logger.warning(f"Job {job_id}: Failed to check disk space: {e}")
        return {'available_gb': 0, 'warning': True, 'critical': True, 'path': path}

class DownloadTracker:
    """Track download results in a thread-safe way."""
    def __init__(self, job_id: int = 1):
        self.job_id = job_id
        self.found_objects = []
        self.modified_objects = []
        self.missing_objects = []
        self.new_objects = []
        self.download_paths = {}
        self.errors_encountered = []
        self.problematic_objects = []
    
    def serialize_metadata(self, metadata: Dict[Hashable, Any]) -> Dict:
        """Safely serialize metadata to ensure it's picklable."""
        simple_metadata = {}
        for k, v in metadata.items():
            try:
                key_str = str(k)
                if hasattr(v, '__dict__'):
                    value_str = str(v.__dict__) if hasattr(v, '__dict__') else str(v)
                elif hasattr(v, 'read'):
                    value_str = f"<file-like object: {type(v).__name__}>"
                else:
                    value_str = str(v)
                simple_metadata[key_str] = value_str
            except Exception:
                simple_metadata[str(k)] = f"<unserializable: {type(v).__name__}>"
        return simple_metadata

# Global instances (will be initialized with job ID)
tracker = None
logger = None
checkpoint_manager = None
progress_tracker = None

def safe_handle_found_object(local_path: str, file_identifier: str, sha256: str, metadata: Dict[Hashable, Any]) -> None:
    """Called when an object is successfully found and downloaded."""
    try:
        simple_metadata = tracker.serialize_metadata(metadata)
        tracker.found_objects.append({
            'local_path': str(local_path),
            'file_identifier': str(file_identifier),
            'sha256': str(sha256),
            'metadata': simple_metadata,
            'timestamp': time.time(),
            'job_id': tracker.job_id
        })
        tracker.download_paths[str(file_identifier)] = str(local_path)
        
        # Mark as completed in checkpoint
        if checkpoint_manager:
            checkpoint_manager.mark_completed(file_identifier)
        
        logger.info(f"✓ Successfully downloaded: {file_identifier}")
    except Exception as e:
        logger.error(f"Error in handle_found_object for {file_identifier}: {e}")
        tracker.errors_encountered.append({
            'type': 'found_object_callback',
            'file_identifier': str(file_identifier),
            'error': str(e),
            'job_id': tracker.job_id
        })

def safe_handle_modified_object(local_path: str, file_identifier: str, new_sha256: str, old_sha256: str, metadata: Dict[Hashable, Any]) -> None:
    """Called when a modified object is found and downloaded."""
    try:
        simple_metadata = tracker.serialize_metadata(metadata)
        tracker.modified_objects.append({
            'local_path': str(local_path),
            'file_identifier': str(file_identifier),
            'new_sha256': str(new_sha256),
            'old_sha256': str(old_sha256),
            'metadata': simple_metadata,
            'timestamp': time.time(),
            'job_id': tracker.job_id
        })
        tracker.download_paths[str(file_identifier)] = str(local_path)
        
        # Mark as completed in checkpoint
        if checkpoint_manager:
            checkpoint_manager.mark_completed(file_identifier)
        
        logger.warning(f"⚠ Modified object downloaded: {file_identifier}")
    except Exception as e:
        logger.error(f"Error in handle_modified_object for {file_identifier}: {e}")

def safe_handle_missing_object(file_identifier: str, sha256: str, metadata: Dict[Hashable, Any]) -> None:
    """Called when a specified object cannot be found."""
    try:
        simple_metadata = tracker.serialize_metadata(metadata)
        tracker.missing_objects.append({
            'file_identifier': str(file_identifier),
            'sha256': str(sha256),
            'metadata': simple_metadata,
            'timestamp': time.time(),
            'job_id': tracker.job_id
        })
        
        # Mark as failed in checkpoint
        if checkpoint_manager:
            checkpoint_manager.mark_failed(file_identifier)
        
        logger.error(f"✗ Missing object: {file_identifier}")
    except Exception as e:
        logger.error(f"Error in handle_missing_object for {file_identifier}: {e}")

def safe_handle_new_object(local_path: str, file_identifier: str, sha256: str, metadata: Dict[Hashable, Any]) -> None:
    """Called when a new object is found (GitHub specific)."""
    try:
        simple_metadata = tracker.serialize_metadata(metadata)
        tracker.new_objects.append({
            'local_path': str(local_path),
            'file_identifier': str(file_identifier),
            'sha256': str(sha256),
            'metadata': simple_metadata,
            'timestamp': time.time(),
            'job_id': tracker.job_id
        })
        
        # Mark as completed in checkpoint
        if checkpoint_manager:
            checkpoint_manager.mark_completed(file_identifier)
        
        logger.info(f"+ New object found: {file_identifier}")
    except Exception as e:
        logger.error(f"Error in handle_new_object for {file_identifier}: {e}")

def download_batch(batch_df, batch_num, total_batches, download_dir, save_repo_format=None, use_multiprocessing=True):
    """Download a batch of objects with error handling."""
    logger.info(f"Processing batch {batch_num}/{total_batches} with {len(batch_df)} objects")
    
    try:
        if use_multiprocessing:
            logger.info(f"  Attempting multiprocess download for batch {batch_num}")
            result = oxl.download_objects(
                objects=batch_df,
                download_dir=download_dir,
                handle_found_object=safe_handle_found_object,
                handle_modified_object=safe_handle_modified_object,
                handle_missing_object=safe_handle_missing_object,
                handle_new_object=safe_handle_new_object,
                save_repo_format=save_repo_format,
                processes=mp.cpu_count(),
            )
        else:
            logger.info(f"  Using single-process download for batch {batch_num}")
            result = oxl.download_objects(
                objects=batch_df,
                download_dir=download_dir,
                handle_found_object=safe_handle_found_object,
                handle_modified_object=safe_handle_modified_object,
                handle_missing_object=safe_handle_missing_object,
                handle_new_object=safe_handle_new_object,
                save_repo_format=save_repo_format,
                processes=1,  # Single process to avoid pickling issues
            )
        
        logger.info(f"  Batch {batch_num} completed successfully")
        return True, None
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"  Batch {batch_num} failed: {error_msg}")
        
        # Check if it's a pickling error
        if "pickle" in error_msg.lower() or "BufferedReader" in error_msg:
            logger.warning(f"  Pickling error detected in batch {batch_num}, will retry with single process")
            return False, "pickling_error"
        else:
            return False, error_msg

def download_objaverse_xl_robust(
    download_dir: str = "./objaverse_xl",
    file_types: List[str] = ['glb', 'obj', 'dae', 'gltf', 'stl'],
    max_objects: Optional[int] = None,
    save_repo_format: Optional[str] = None,
    temp_dir: str = None,
    batch_size: int = 1000,
    start_offset: int = 0,
    job_id: int = 1,
    cleanup_temp: bool = True,
    min_disk_space_gb: float = 10.0,
    enable_checkpoints: bool = True
) -> Dict:
    """
    Download Objaverse XL dataset with robust batch processing and error recovery.
    
    Args:
        download_dir: Directory to download objects to (permanent storage)
        file_types: List of file types to download
        max_objects: Maximum number of objects to download
        save_repo_format: Format to save GitHub repos
        temp_dir: Custom temporary directory (will be cleaned regularly)
        batch_size: Number of objects to process in each batch
        start_offset: Starting index for this job (for job arrays)
        job_id: Job ID for distinguishing parallel jobs
        cleanup_temp: Whether to clean temp directory during and after download
        min_disk_space_gb: Minimum disk space threshold in GB
        enable_checkpoints: Whether to enable checkpoint/resume functionality
    
    Returns:
        Dictionary with download results
    """
    global tracker, checkpoint_manager, progress_tracker
    tracker = DownloadTracker(job_id)  # Initialize tracker with job ID
    
    # Initialize checkpoint manager if enabled
    if enable_checkpoints:
        checkpoint_manager = CheckpointManager(job_id)
        logger.info(f"Job {job_id}: Checkpoint/resume enabled")
    
    # Set up custom temporary directory
    if temp_dir is None:
        temp_dir = os.path.expanduser(f"~/objaverse_temp/job_{job_id}")
    temp_dir = setup_custom_temp_dir(temp_dir, job_id)
    
    # Initial disk space check
    disk_status = check_disk_space(temp_dir, min_disk_space_gb, job_id)
    if disk_status['critical']:
        error_msg = f"Critical disk space issue: Only {disk_status['available_gb']} GB available"
        logger.error(f"Job {job_id}: {error_msg}")
        return {'error': error_msg}
    
    # Clean temp directory at the start to remove any leftover files from previous runs
    if cleanup_temp:
        logger.info(f"Job {job_id}: Cleaning temp directory before starting (removing leftover files)")
        cleanup_temp_files(temp_dir, aggressive=True)
    
    logger.info(f"Job {job_id}: Setting up git LFS with GitHub token")
    setup_git_lfs_with_token()
    
    logger.info(f"Job {job_id}: Loading Objaverse XL annotations")
    try:
        annotations = oxl.get_annotations()
        
        logger.info(f"Job {job_id}: Source distribution: {annotations['source'].value_counts().to_dict()}")
        annotations = annotations[annotations['source'] != 'github']
        logger.info(f"Job {job_id}: After filtering GitHub: {len(annotations)} objects remaining")
        
    except Exception as e:
        logger.error(f"Job {job_id}: Failed to load annotations: {e}")
        return {'error': str(e)}
    
    # Filter by file types
    if file_types:
        filtered = annotations[annotations['fileType'].isin(file_types)]
        logger.info(f"Job {job_id}: Filtered to {len(filtered)} objects with file types: {file_types}")
    else:
        filtered = annotations
        logger.info(f"Job {job_id}: Using all {len(filtered)} objects")
    
    

    if len(filtered) == 0:
        logger.warning(f"Job {job_id}: No objects found with file types: {file_types}")
        return {'error': f'No objects found with file types: {file_types}'}
    
    # Apply start offset and limit for this job
    total_available = len(filtered)
    end_offset = start_offset + max_objects if max_objects else total_available
    
    if start_offset >= total_available:
        logger.warning(f"Job {job_id}: Start offset {start_offset} exceeds available objects {total_available}")
        return {'error': f'Start offset {start_offset} exceeds available objects {total_available}'}
    
    # Slice the dataset for this job
    job_subset = filtered.iloc[start_offset:min(end_offset, total_available)]
    
    # Filter out already completed objects if checkpoints are enabled
    if enable_checkpoints and len(checkpoint_manager.completed_objects) > 0:
        logger.info(f"Job {job_id}: Filtering out {len(checkpoint_manager.completed_objects)} already completed objects")
        job_subset = job_subset[~job_subset['fileIdentifier'].isin(checkpoint_manager.completed_objects)]
        logger.info(f"Job {job_id}: {len(job_subset)} objects remaining to download")
    
    logger.info(f"Job {job_id}: Processing objects {start_offset} to {min(end_offset, total_available)-1}")
    logger.info(f"Job {job_id}: Total objects for this job: {len(job_subset)}")
    
    # Split into batches
    total_objects = len(job_subset)
    
    if total_objects == 0:
        logger.info(f"Job {job_id}: All objects already completed, nothing to download")
        return {
            'success': True,
            'summary': {
                'job_id': job_id,
                'message': 'All objects already completed',
                'total_requested': 0
            }
        }
    
    num_batches = (total_objects + batch_size - 1) // batch_size
    
    # Initialize progress tracker
    progress_tracker = ProgressTracker(job_id, total_objects)
    progress_tracker.total_batches = num_batches
    
    logger.info(f"Job {job_id}: Starting download of {total_objects} objects in {num_batches} batches")
    logger.info(f"Job {job_id}: Batch size: {batch_size} objects")
    logger.info(f"Job {job_id}: ====== DIRECTORY CONFIGURATION ======")
    logger.info(f"Job {job_id}: TEMP directory (will be cleaned): {temp_dir}")
    logger.info(f"Job {job_id}: DOWNLOAD directory (permanent): {download_dir}")
    logger.info(f"Job {job_id}: OUTPUT reports directory: ./output/job_{job_id}")
    logger.info(f"Job {job_id}: CHECKPOINT directory: ./checkpoints")
    if cleanup_temp:
        logger.info(f"Job {job_id}: CLEANUP: Enabled (before/during/after)")
    else:
        logger.info(f"Job {job_id}: CLEANUP: Disabled (temp files will accumulate)")
    logger.info(f"Job {job_id}: ======================================")
    
    start_time = time.time()
    failed_batches = []
    last_disk_check_time = time.time()
    disk_check_interval = 300  # Check every 5 minutes
    
    # Process each batch
    for batch_num in range(num_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, total_objects)
        batch_df = job_subset.iloc[start_idx:end_idx]
        
        # Update progress tracker
        progress_tracker.update(batch_num=batch_num + 1)
        
        # Periodic disk space check
        current_time = time.time()
        if current_time - last_disk_check_time > disk_check_interval:
            disk_status = check_disk_space(temp_dir, min_disk_space_gb, job_id)
            if disk_status['critical']:
                logger.error(f"Job {job_id}: STOPPING - Critical disk space at batch {batch_num + 1}")
                failed_batches.append((batch_num + 1, "critical_disk_space"))
                break
            last_disk_check_time = current_time
        
        # Try with multiprocessing first
        success, error_type = download_batch(
            batch_df, 
            batch_num + 1, 
            num_batches,
            download_dir,
            save_repo_format,
            use_multiprocessing=True
        )
        
        # Update progress with batch results
        batch_successful = len([o for o in tracker.found_objects if o.get('timestamp', 0) > start_time])
        batch_failed = len([o for o in tracker.missing_objects if o.get('timestamp', 0) > start_time])
        progress_tracker.update(successful=len(batch_df) if success else 0, 
                               failed=0 if success else len(batch_df))
        
        # If failed due to pickling, retry with single process
        if not success and error_type == "pickling_error":
            logger.info(f"Job {job_id}: Retrying batch {batch_num + 1} with single process due to pickling error")
            success, error_type = download_batch(
                batch_df,
                batch_num + 1,
                num_batches,
                download_dir,
                save_repo_format,
                use_multiprocessing=False
            )
            
            if not success:
                # If still failing, try individual objects
                logger.warning(f"Job {job_id}: Batch {batch_num + 1} still failing, trying individual objects")
                for idx, row in batch_df.iterrows():
                    try:
                        single_df = pd.DataFrame([row])
                        oxl.download_objects(
                            objects=single_df,
                            download_dir=download_dir,
                            handle_found_object=safe_handle_found_object,
                            handle_modified_object=safe_handle_modified_object,
                            handle_missing_object=safe_handle_missing_object,
                            handle_new_object=safe_handle_new_object,
                            save_repo_format=save_repo_format,
                            processes=1,
                        )
                    except Exception as e:
                        logger.error(f"Job {job_id}: Failed to download object {row.get('fileIdentifier', 'unknown')}: {e}")
                        tracker.problematic_objects.append({
                            'file_identifier': str(row.get('fileIdentifier', 'unknown')),
                            'error': str(e),
                            'row_data': row.to_dict(),
                            'job_id': job_id
                        })
        
        elif not success:
            failed_batches.append((batch_num + 1, error_type))
        
        # Save checkpoint after each batch
        if enable_checkpoints and checkpoint_manager:
            checkpoint_manager.save_checkpoint()
        
        # Save progress periodically
        progress_tracker.save_progress()
        
        # Log progress every 10 batches
        if (batch_num + 1) % 10 == 0:
            progress_tracker.log_progress()
        
        # Clean up temp files after every batch to avoid filling up temp space
        if cleanup_temp:
            cleanup_temp_files(temp_dir, aggressive=False)
    
    # Final progress save
    progress_tracker.save_progress(force=True)
    progress_tracker.log_progress()
    
    # Final aggressive cleanup - remove everything from temp directory
    if cleanup_temp:
        logger.info(f"Job {job_id}: Performing final cleanup of temporary directory")
        cleanup_temp_files(temp_dir, aggressive=True)
    else:
        logger.info(f"Job {job_id}: Skipping temp cleanup (--no-cleanup-temp flag set)")
    
    # Calculate summary
    duration = time.time() - start_time
    total_requested = total_objects
    
    summary = {
        'job_id': job_id,
        'start_offset': start_offset,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'duration_seconds': round(duration, 2),
        'total_requested': total_requested,
        'found_objects': len(tracker.found_objects),
        'modified_objects': len(tracker.modified_objects),
        'missing_objects': len(tracker.missing_objects),
        'new_objects': len(tracker.new_objects),
        'problematic_objects': len(tracker.problematic_objects),
        'errors_encountered': len(tracker.errors_encountered),
        'failed_batches': len(failed_batches),
        'success_rate': len(tracker.found_objects) / total_requested * 100 if total_requested > 0 else 0,
        'objects_per_second': len(tracker.found_objects) / duration if duration > 0 else 0,
        'checkpoint_enabled': enable_checkpoints,
        'resumed_from_checkpoint': enable_checkpoints and len(checkpoint_manager.completed_objects) > 0 if checkpoint_manager else False
    }
    
    # Save metadata and reports with job ID
    output_dir = f"./output/job_{job_id}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Export processed subset to CSV
    metadata_file = os.path.join(output_dir, f"objaverse_xl_metadata_job_{job_id}.csv")
    try:
        job_subset.to_csv(metadata_file, index=False)
        logger.info(f"Job {job_id}: Saved metadata to {metadata_file}")
    except Exception as e:
        logger.error(f"Job {job_id}: Failed to save metadata: {e}")
    
    # Save detailed report
    report_file = os.path.join(output_dir, f"download_report_job_{job_id}.json")
    detailed_report = {
        'summary': summary,
        'found_objects': tracker.found_objects,
        'modified_objects': tracker.modified_objects,
        'missing_objects': tracker.missing_objects,
        'new_objects': tracker.new_objects,
        'problematic_objects': tracker.problematic_objects,
        'download_paths': tracker.download_paths,
        'errors_encountered': tracker.errors_encountered,
        'failed_batches': failed_batches
    }
    
    try:
        with open(report_file, 'w') as f:
            json.dump(detailed_report, f, indent=4, default=str)
        logger.info(f"Job {job_id}: Detailed download report saved to {report_file}")
    except Exception as e:
        logger.error(f"Job {job_id}: Failed to save report: {e}")
    
    # Save problematic objects separately for debugging
    if tracker.problematic_objects:
        problematic_file = os.path.join(output_dir, f"problematic_objects_job_{job_id}.json")
        try:
            with open(problematic_file, 'w') as f:
                json.dump(tracker.problematic_objects, f, indent=4, default=str)
            logger.warning(f"Job {job_id}: Saved {len(tracker.problematic_objects)} problematic objects to {problematic_file}")
        except Exception as e:
            logger.error(f"Job {job_id}: Failed to save problematic objects: {e}")
    
    # Log summary
    logger.info(f"Job {job_id}: Download Summary:")
    logger.info(f"  Total requested: {summary['total_requested']}")
    logger.info(f"  Successfully downloaded: {summary['found_objects']}")
    logger.info(f"  Modified objects: {summary['modified_objects']}")
    logger.info(f"  Missing objects: {summary['missing_objects']}")
    logger.info(f"  New objects found: {summary['new_objects']}")
    logger.info(f"  Problematic objects: {summary['problematic_objects']}")
    logger.info(f"  Errors encountered: {summary['errors_encountered']}")
    logger.info(f"  Failed batches: {summary['failed_batches']}")
    logger.info(f"  Success rate: {summary['success_rate']:.2f}%")
    logger.info(f"  Download speed: {summary['objects_per_second']:.2f} objects/sec")
    
    if enable_checkpoints:
        logger.info(f"  Checkpoint saved with {len(checkpoint_manager.completed_objects)} completed objects")
    
    # Log any problematic objects
    if tracker.problematic_objects:
        logger.warning(f"Job {job_id}: Encountered {len(tracker.problematic_objects)} problematic objects:")
        for obj in tracker.problematic_objects[:5]:  # Show first 5
            logger.warning(f"  - {obj['file_identifier']}: {obj['error']}")
        if len(tracker.problematic_objects) > 5:
            logger.warning(f"  ... and {len(tracker.problematic_objects) - 5} more")
    
    return {
        'success': True,
        'summary': summary,
        'download_paths': tracker.download_paths,
        'detailed_report': detailed_report
    }

def download_objaverse_regular(num_objects: int = 100000, temp_dir: str = None, start_offset: int = 0, job_id: int = 1):
    """Download regular Objaverse dataset with custom temp directory and job support."""
    # Set up custom temporary directory
    if temp_dir is None:
        temp_dir = os.path.expanduser(f"~/objaverse_temp/job_{job_id}")
    setup_custom_temp_dir(temp_dir, job_id)
    
    logger.info(f"Job {job_id}: Downloading {num_objects} objects from Objaverse starting at offset {start_offset}")
    
    try:
        uids = objaverse.load_uids()
        total_available = len(uids)
        
        if start_offset >= total_available:
            logger.warning(f"Job {job_id}: Start offset {start_offset} exceeds available objects {total_available}")
            return
        
        # Calculate end offset
        end_offset = min(start_offset + num_objects, total_available)
        
        # Get subset of UIDs for this job
        job_uids = uids[start_offset:end_offset]
        actual_objects = len(job_uids)
        
        logger.info(f"Job {job_id}: Processing {actual_objects} objects from offset {start_offset} to {end_offset-1}")

        # Load metadata for these objects
        annotations = objaverse.load_annotations(job_uids)
        
        # Try with single process first (more stable)
        try:
            objects = objaverse.load_objects(
                uids=job_uids,
                download_processes=1)  # Use single process to avoid issues
        except Exception as e:
            logger.warning(f"Job {job_id}: Single process download failed: {e}, trying with multiprocessing")
            objects = objaverse.load_objects(
                uids=job_uids,
                download_processes=mp.cpu_count())
        
        # save the uids and file paths to a JSON file
        output_dir = f"./output/job_{job_id}"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"objaverse_metadata_job_{job_id}.json")
        with open(output_file, 'w') as f:
            json.dump({
                "job_id": job_id,
                "start_offset": start_offset,
                "objects_processed": actual_objects,
                "objects": objects
            }, f, indent=4)
        logger.info(f"Job {job_id}: Saved metadata to {output_file}")
        
    except Exception as e:
        logger.error(f"Job {job_id}: Failed to download regular Objaverse: {e}")
        raise

def download_objaverse(
    xl: bool = False,
    num_objects: int = 10000,
    file_types: List[str] = ['glb', 'obj', 'dae', 'gltf', 'stl'],
    save_repos: bool = True,
    temp_dir: str = None,
    batch_size: int = 1000,
    start_offset: int = 0,
    job_id: int = 1,
    download_dir: str = "./objaverse_xl",
    cleanup_temp: bool = True,
    min_disk_space_gb: float = 10.0,
    enable_checkpoints: bool = True
):
    """
    Main download function with custom temporary directory support and job array functionality.
    
    Args:
        xl: Whether to download Objaverse XL dataset
        num_objects: Number of objects to download
        file_types: List of file types to download for XL dataset
        save_repos: Whether to save GitHub repositories (XL only)
        temp_dir: Custom temporary directory path (will be cleaned if cleanup_temp=True)
        batch_size: Number of objects per batch for XL downloads
        start_offset: Starting index for this job (for job arrays)
        job_id: Job ID for distinguishing parallel jobs
        download_dir: Directory to download objects to (permanent storage)
        cleanup_temp: Whether to clean temp directory during and after download
        min_disk_space_gb: Minimum disk space threshold in GB
        enable_checkpoints: Whether to enable checkpoint/resume functionality
    """
    if xl:
        logger.info(f"Job {job_id}: Starting Objaverse XL download with robust batch processing")
        logger.info(f"Job {job_id}: Processing {num_objects} objects starting from offset {start_offset}")
        save_repo_format = "files" if save_repos else None
        max_objects = num_objects if num_objects < 100000 else None
        
        results = download_objaverse_xl_robust(
            download_dir=download_dir,
            file_types=file_types,
            max_objects=max_objects,
            save_repo_format=save_repo_format,
            temp_dir=temp_dir,
            batch_size=batch_size,
            start_offset=start_offset,
            job_id=job_id,
            cleanup_temp=cleanup_temp,
            min_disk_space_gb=min_disk_space_gb,
            enable_checkpoints=enable_checkpoints
        )
        
        if 'error' in results:
            logger.error(f"Job {job_id}: XL Download failed: {results['error']}")
            return False
        
        logger.info(f"Job {job_id}: XL Download completed successfully!")
        return True
    else:
        # Use updated implementation for regular Objaverse
        try:
            download_objaverse_regular(num_objects, temp_dir, start_offset, job_id)
            return True
        except Exception as e:
            logger.error(f"Job {job_id}: Regular Objaverse download failed: {e}")
            return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download Objaverse dataset with robust error handling and job array support.")
    parser.add_argument(
        "--xl",
        action="store_true",
        default=True,
        help="Download the Objaverse XL dataset instead of the regular Objaverse dataset.",
    )
    parser.add_argument(
        "--num-objects",
        type=int,
        default=1000000,
        help="Number of objects to download. Default: 1000000",
    )
    parser.add_argument(
        "--file-types",
        nargs='+',
        default=['glb', 'obj', 'stl'],
        help="File types to download for XL dataset (e.g., glb obj fbx). Default: glb",
    )
    parser.add_argument(
        "--save-repos",
        action="store_true",
        default=True,
        help="Save GitHub repositories as files (XL only). Default: True",
    )
    parser.add_argument(
        "--temp-dir",
        type=str,
        default=None,
        help="Custom temporary directory path. Default: ~/objaverse_temp/job_X",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of objects per batch for XL downloads. Default: 1000",
    )
    parser.add_argument(
        "--start-offset",
        type=int,
        default=0,
        help="Starting index for this job (for job arrays). Default: 0",
    )
    parser.add_argument(
        "--job-id",
        type=int,
        default=1,
        help="Job ID for distinguishing parallel jobs. Default: 1",
    )
    parser.add_argument(
        "--download-dir",
        type=str,
        default="./objaverse_xl",
        help="Directory to download objects to. Default: ./objaverse_xl",
    )
    parser.add_argument(
        "--no-cleanup-temp",
        action="store_true",
        default=False,
        help="Skip cleaning temporary directory (useful for debugging). Default: False (cleanup enabled)",
    )
    parser.add_argument(
        "--manual-cleanup",
        action="store_true",
        default=False,
        help="Manually clean all temp directories and exit (useful after crashes)",
    )
    parser.add_argument(
        "--cleanup-temp-dir",
        type=str,
        default=None,
        help="Custom base temp directory for manual cleanup. Default: ~/objaverse_temp",
    )
    parser.add_argument(
        "--min-disk-space-gb",
        type=float,
        default=10.0,
        help="Minimum disk space threshold in GB. Default: 10.0",
    )
    parser.add_argument(
        "--no-checkpoints",
        action="store_true",
        default=False,
        help="Disable checkpoint/resume functionality. Default: False (checkpoints enabled)",
    )
    
    args = parser.parse_args()
    
    # Handle manual cleanup if requested
    if args.manual_cleanup:
        manual_cleanup_all_temp_dirs(args.cleanup_temp_dir)
        exit(0)
    
    # Initialize logging with job ID
    logger = setup_logging(args.job_id)
    
    try:
        success = download_objaverse(
            xl=args.xl,
            num_objects=args.num_objects,
            file_types=args.file_types,
            save_repos=args.save_repos,
            temp_dir=args.temp_dir,
            batch_size=args.batch_size,
            start_offset=args.start_offset,
            job_id=args.job_id,
            download_dir=args.download_dir,
            cleanup_temp=not args.no_cleanup_temp,
            min_disk_space_gb=args.min_disk_space_gb,
            enable_checkpoints=not args.no_checkpoints
        )
        exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info(f"Job {args.job_id}: Download interrupted by user")
        # Save final checkpoint before exiting
        if checkpoint_manager:
            checkpoint_manager.save_checkpoint()
            logger.info(f"Job {args.job_id}: Checkpoint saved before exit")
        exit(1)
    except Exception as e:
        logger.error(f"Job {args.job_id}: Unexpected error: {e}")
        # Save final checkpoint before exiting
        if checkpoint_manager:
            checkpoint_manager.save_checkpoint()
            logger.info(f"Job {args.job_id}: Checkpoint saved after error")
        exit(1)