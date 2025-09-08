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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('objaverse_download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration

dotenv.load_dotenv()  # Load environment variables from .env file if present
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
os.environ["GITHUB_TOKEN"] = GITHUB_TOKEN
os.environ["Github_token"] = GITHUB_TOKEN

def setup_custom_temp_dir(base_dir: str = None):
    """Set up custom temporary directory with more space."""
    if base_dir is None:
        # Use home directory or current working directory
        base_dir = os.path.expanduser("~/temp") if os.path.exists(os.path.expanduser("~")) else "./temp"
    
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
    def __init__(self):
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

# Global tracker instance
tracker = DownloadTracker()

def safe_handle_found_object(local_path: str, file_identifier: str, sha256: str, metadata: Dict[Hashable, Any]) -> None:
    """Called when an object is successfully found and downloaded."""
    try:
        simple_metadata = tracker.serialize_metadata(metadata)
        tracker.found_objects.append({
            'local_path': str(local_path),
            'file_identifier': str(file_identifier),
            'sha256': str(sha256),
            'metadata': simple_metadata,
            'timestamp': time.time()
        })
        tracker.download_paths[str(file_identifier)] = str(local_path)
        logger.info(f"✓ Successfully downloaded: {file_identifier}")
    except Exception as e:
        logger.error(f"Error in handle_found_object for {file_identifier}: {e}")
        tracker.errors_encountered.append({
            'type': 'found_object_callback',
            'file_identifier': str(file_identifier),
            'error': str(e)
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
            'timestamp': time.time()
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
            'timestamp': time.time()
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
            'timestamp': time.time()
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
    file_types: List[str] = ['glb'],
    max_objects: Optional[int] = None,
    save_repo_format: Optional[str] = None,
    temp_dir: str = None,
    batch_size: int = 1000
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
    
    Returns:
        Dictionary with download results
    """
    global tracker
    tracker = DownloadTracker()  # Reset tracker
    
    # Set up custom temporary directory
    if temp_dir is None:
        temp_dir = os.path.expanduser("~/objaverse_temp")
    temp_dir = setup_custom_temp_dir(temp_dir)
    
    logger.info("Setting up git LFS with GitHub token")
    setup_git_lfs_with_token()
    
    logger.info("Loading Objaverse XL annotations")
    try:
        annotations = oxl.get_annotations()
    except Exception as e:
        logger.error(f"Failed to load annotations: {e}")
        return {'error': str(e)}
    
    # Filter by file types
    if file_types:
        filtered = annotations[annotations['fileType'].isin(file_types)]
        logger.info(f"Filtered to {len(filtered)} objects with file types: {file_types}")
    else:
        filtered = annotations
        logger.info(f"Using all {len(filtered)} objects")
    
    if len(filtered) == 0:
        logger.warning(f"No objects found with file types: {file_types}")
        return {'error': f'No objects found with file types: {file_types}'}
    
    # Limit number of objects if specified
    if max_objects and len(filtered) > max_objects:
        filtered = filtered.sample(n=max_objects, random_state=42)
        logger.info(f"Randomly sampled {max_objects} objects")
    
    # Split into batches
    total_objects = len(filtered)
    num_batches = (total_objects + batch_size - 1) // batch_size
    
    logger.info(f"Starting download of {total_objects} objects in {num_batches} batches")
    logger.info(f"Batch size: {batch_size} objects")
    logger.info(f"Using temporary directory: {temp_dir}")
    
    start_time = time.time()
    failed_batches = []
    
    # Process each batch
    for batch_num in range(num_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, total_objects)
        batch_df = filtered.iloc[start_idx:end_idx]
        
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
            logger.info(f"Retrying batch {batch_num + 1} with single process due to pickling error")
            success, error_type = download_batch(
                batch_df,
                batch_num + 1,
                num_batches,
                save_repo_format,
                use_multiprocessing=False
            )
            
            if not success:
                # If still failing, try individual objects
                logger.warning(f"Batch {batch_num + 1} still failing, trying individual objects")
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
                        logger.error(f"Failed to download object {row.get('fileIdentifier', 'unknown')}: {e}")
                        tracker.problematic_objects.append({
                            'file_identifier': str(row.get('fileIdentifier', 'unknown')),
                            'error': str(e),
                            'row_data': row.to_dict()
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
    
    # Save metadata and reports
    output_dir = "./"
    
    # Export filtered annotations to CSV
    metadata_file = os.path.join(output_dir, "objaverse_xl_metadata.csv")
    try:
        filtered.to_csv(metadata_file, index=False)
        logger.info(f"Saved metadata to {metadata_file}")
    except Exception as e:
        logger.error(f"Failed to save metadata: {e}")
    
    # Save detailed report
    report_file = os.path.join(output_dir, "download_report.json")
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
        logger.info(f"Detailed download report saved to {report_file}")
    except Exception as e:
        logger.error(f"Failed to save report: {e}")
    
    # Save problematic objects separately for debugging
    if tracker.problematic_objects:
        problematic_file = os.path.join(output_dir, "problematic_objects.json")
        try:
            with open(problematic_file, 'w') as f:
                json.dump(tracker.problematic_objects, f, indent=4, default=str)
            logger.warning(f"Saved {len(tracker.problematic_objects)} problematic objects to {problematic_file}")
        except Exception as e:
            logger.error(f"Failed to save problematic objects: {e}")
    
    # Log summary
    logger.info(f"Download Summary:")
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
        logger.warning(f"Encountered {len(tracker.problematic_objects)} problematic objects:")
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

def download_objaverse_regular(num_objects: int = 100000, temp_dir: str = None):
    """Download regular Objaverse dataset with custom temp directory."""
    # Set up custom temporary directory
    if temp_dir is None:
        temp_dir = os.path.expanduser("~/objaverse_temp")
    setup_custom_temp_dir(temp_dir)
    
    logger.info(f"Downloading {num_objects} random objects from Objaverse")
    
    try:
        uids = objaverse.load_uids()
        if num_objects > len(uids):
            logger.warning(f"Requested {num_objects} objects, but only {len(uids)} available. Downloading all.")
            num_objects = len(uids)

        random_object_uids = random.sample(uids, num_objects)
        
        # Load metadata for these objects
        annotations = objaverse.load_annotations(random_object_uids)
        
        # Try with single process first (more stable)
        try:
            objects = objaverse.load_objects(
                uids=random_object_uids,
                download_processes=1)  # Use single process to avoid issues
        except Exception as e:
            logger.warning(f"Single process download failed: {e}, trying with multiprocessing")
            objects = objaverse.load_objects(
                uids=random_object_uids,
                download_processes=mp.cpu_count())
        
        # save the uids and file paths to a JSON file
        output_dir = "./"
        output_file = os.path.join(output_dir, "objaverse_metadata.json")
        with open(output_file, 'w') as f:
            json.dump({
                "objects": objects
            }, f, indent=4)
        logger.info(f"Saved metadata to {output_file}")
        
    except Exception as e:
        logger.error(f"Failed to download regular Objaverse: {e}")
        raise

def download_objaverse(
    xl: bool = False,
    num_objects: int = 10000,
    file_types: List[str] = ['glb'],
    save_repos: bool = True,
    temp_dir: str = None,
    batch_size: int = 1000
):
    """
    Main download function with custom temporary directory support.
    
    Args:
        xl: Whether to download Objaverse XL dataset
        num_objects: Number of objects to download
        file_types: List of file types to download for XL dataset
        save_repos: Whether to save GitHub repositories (XL only)
        temp_dir: Custom temporary directory path
        batch_size: Number of objects per batch for XL downloads
    """
    if xl:
        logger.info("Starting Objaverse XL download with robust batch processing")
        save_repo_format = "files" if save_repos else None
        max_objects = num_objects if num_objects < 100000 else None
        
        results = download_objaverse_xl_robust(
            file_types=file_types,
            max_objects=max_objects,
            save_repo_format=save_repo_format,
            temp_dir=temp_dir,
            batch_size=batch_size
        )
        
        if 'error' in results:
            logger.error(f"XL Download failed: {results['error']}")
            return False
        
        logger.info("XL Download completed successfully!")
        return True
    else:
        # Use updated implementation for regular Objaverse
        try:
            download_objaverse_regular(num_objects, temp_dir)
            return True
        except Exception as e:
            logger.error(f"Regular Objaverse download failed: {e}")
            return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download Objaverse dataset with robust error handling.")
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
        default=['glb'],
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
        help="Custom temporary directory path. Default: ~/objaverse_temp",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of objects per batch for XL downloads. Default: 1000",
    )
    
    args = parser.parse_args()
    
    try:
        success = download_objaverse(
            xl=args.xl,
            num_objects=args.num_objects,
            file_types=args.file_types,
            save_repos=False,
            temp_dir=args.temp_dir,
            batch_size=args.batch_size
        )
        exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("Download interrupted by user")
        exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        exit(1)