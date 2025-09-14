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
def setup_logging(job_id: int = 1):
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

def cleanup_temp_files(temp_dir: str):
    """Clean up temporary files periodically."""
    try:
        import shutil
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
                
        logger.info(f"Cleaned up {len(temp_files)} temporary files")
    except Exception as e:
        logger.warning(f"Error cleaning temp files: {e}")

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

# Global tracker instance (will be initialized with job ID)
tracker = None
logger = None

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
        logger.info(f"+ New object found: {file_identifier}")
    except Exception as e:
        logger.error(f"Error in handle_new_object for {file_identifier}: {e}")

def download_batch(batch_df, batch_num, total_batches, save_repo_format=None, use_multiprocessing=True):
    """Download a batch of objects with error handling."""
    logger.info(f"Processing batch {batch_num}/{total_batches} with {len(batch_df)} objects")
    
    try:
        if use_multiprocessing:
            logger.info(f"  Attempting multiprocess download for batch {batch_num}")
            result = oxl.download_objects(
                objects=batch_df,
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
    job_id: int = 1
) -> Dict:
    """
    Download Objaverse XL dataset with robust batch processing and error recovery.
    
    Args:
        download_dir: Directory to download objects to
        file_types: List of file types to download
        max_objects: Maximum number of objects to download
        save_repo_format: Format to save GitHub repos
        temp_dir: Custom temporary directory
        batch_size: Number of objects to process in each batch
        start_offset: Starting index for this job (for job arrays)
        job_id: Job ID for distinguishing parallel jobs
    
    Returns:
        Dictionary with download results
    """
    global tracker
    tracker = DownloadTracker(job_id)  # Initialize tracker with job ID
    
    # Set up custom temporary directory
    if temp_dir is None:
        temp_dir = os.path.expanduser(f"~/objaverse_temp/job_{job_id}")
    temp_dir = setup_custom_temp_dir(temp_dir, job_id)
    
    logger.info(f"Job {job_id}: Setting up git LFS with GitHub token")
    setup_git_lfs_with_token()
    
    logger.info(f"Job {job_id}: Loading Objaverse XL annotations")
    try:
        annotations = oxl.get_annotations()
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
    logger.info(f"Job {job_id}: Processing objects {start_offset} to {min(end_offset, total_available)-1}")
    logger.info(f"Job {job_id}: Total objects for this job: {len(job_subset)}")
    
    # Split into batches
    total_objects = len(job_subset)
    num_batches = (total_objects + batch_size - 1) // batch_size
    
    logger.info(f"Job {job_id}: Starting download of {total_objects} objects in {num_batches} batches")
    logger.info(f"Job {job_id}: Batch size: {batch_size} objects")
    logger.info(f"Job {job_id}: Using temporary directory: {temp_dir}")
    
    start_time = time.time()
    failed_batches = []
    
    # Process each batch
    for batch_num in range(num_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, total_objects)
        batch_df = job_subset.iloc[start_idx:end_idx]
        
        # Try with multiprocessing first
        success, error_type = download_batch(
            batch_df, 
            batch_num + 1, 
            num_batches, 
            save_repo_format,
            use_multiprocessing=True
        )
        
        # If failed due to pickling, retry with single process
        if not success and error_type == "pickling_error":
            logger.info(f"Job {job_id}: Retrying batch {batch_num + 1} with single process due to pickling error")
            success, error_type = download_batch(
                batch_df,
                batch_num + 1,
                num_batches,
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
        
        # Clean up temp files periodically
        if (batch_num + 1) % 10 == 0:
            cleanup_temp_files(temp_dir)
    
    # Final cleanup
    cleanup_temp_files(temp_dir)
    
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
        'objects_per_second': len(tracker.found_objects) / duration if duration > 0 else 0
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
    job_id: int = 1
):
    """
    Main download function with custom temporary directory support and job array functionality.
    
    Args:
        xl: Whether to download Objaverse XL dataset
        num_objects: Number of objects to download
        file_types: List of file types to download for XL dataset
        save_repos: Whether to save GitHub repositories (XL only)
        temp_dir: Custom temporary directory path
        batch_size: Number of objects per batch for XL downloads
        start_offset: Starting index for this job (for job arrays)
        job_id: Job ID for distinguishing parallel jobs
    """
    if xl:
        logger.info(f"Job {job_id}: Starting Objaverse XL download with robust batch processing")
        logger.info(f"Job {job_id}: Processing {num_objects} objects starting from offset {start_offset}")
        save_repo_format = "files" if save_repos else None
        max_objects = num_objects if num_objects < 100000 else None
        
        results = download_objaverse_xl_robust(
            file_types=file_types,
            max_objects=max_objects,
            save_repo_format=save_repo_format,
            temp_dir=temp_dir,
            batch_size=batch_size,
            start_offset=start_offset,
            job_id=job_id
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
        default=100000,
        help="Number of objects to download. Default: 100000",
    )
    parser.add_argument(
        "--file-types",
        nargs='+',
        default=['glb', 'obj', 'dae', 'gltf', 'stl'],
        help="File types to download for XL dataset (e.g., glb obj fbx). Default: glb",
    )
    parser.add_argument(
        "--save-repos",
        action="store_true",
        default=False,
        help="Save GitHub repositories as files (XL only). Default: False",
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
    
    args = parser.parse_args()
    
    # Initialize logging with job ID
    logger = setup_logging(args.job_id)
    
    try:
        success = download_objaverse(
            xl=args.xl,
            num_objects=args.num_objects,
            file_types=args.file_types,
            save_repos=False,
            temp_dir=args.temp_dir,
            batch_size=args.batch_size,
            start_offset=args.start_offset,
            job_id=args.job_id
        )
        exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info(f"Job {args.job_id}: Download interrupted by user")
        exit(1)
    except Exception as e:
        logger.error(f"Job {args.job_id}: Unexpected error: {e}")
        exit(1)