import random
import multiprocessing as mp
from multiprocessing import Manager
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
import dotenv   
dotenv.load_dotenv()  # Load environment variables from .env file if present
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
# Configuration
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

# Global manager for multiprocessing-safe data structures
manager = None
found_objects = None
modified_objects = None
missing_objects = None
new_objects = None
download_paths = None
errors_encountered = None

def serialize_metadata(metadata: Dict[Hashable, Any]) -> Dict:
    """Safely serialize metadata to ensure it's picklable."""
    simple_metadata = {}
    for k, v in metadata.items():
        try:
            # Try to convert to string
            key_str = str(k)
            if hasattr(v, '__dict__'):
                # If it's an object with attributes, try to extract them
                value_str = str(v.__dict__) if hasattr(v, '__dict__') else str(v)
            elif hasattr(v, 'read'):
                # If it's a file-like object, just note its type
                value_str = f"<file-like object: {type(v).__name__}>"
            else:
                value_str = str(v)
            simple_metadata[key_str] = value_str
        except Exception as e:
            # If we can't serialize it, store a placeholder
            simple_metadata[str(k)] = f"<unserializable: {type(v).__name__}>"
    return simple_metadata

def safe_handle_found_object(
    local_path: str,
    file_identifier: str,
    sha256: str,
    metadata: Dict[Hashable, Any]
) -> None:
    """Called when an object is successfully found and downloaded."""
    try:
        global found_objects, download_paths
        
        # Serialize metadata to ensure it's picklable
        simple_metadata = serialize_metadata(metadata)
        
        found_objects.append({
            'local_path': str(local_path),
            'file_identifier': str(file_identifier),
            'sha256': str(sha256),
            'metadata': simple_metadata,
            'timestamp': time.time()
        })
        download_paths[str(file_identifier)] = str(local_path)
        logger.info(f"✓ Successfully downloaded: {file_identifier}")
        logger.debug(f"  Local path: {local_path}")
    except Exception as e:
        logger.error(f"Error in handle_found_object for {file_identifier}: {e}")
        if errors_encountered is not None:
            errors_encountered.append({
                'type': 'found_object_callback',
                'file_identifier': str(file_identifier),
                'error': str(e)
            })

def safe_handle_modified_object(
    local_path: str,
    file_identifier: str,
    new_sha256: str,
    old_sha256: str,
    metadata: Dict[Hashable, Any]
) -> None:
    """Called when a modified object is found and downloaded."""
    try:
        global modified_objects, download_paths
        
        # Serialize metadata to ensure it's picklable
        simple_metadata = serialize_metadata(metadata)
        
        modified_objects.append({
            'local_path': str(local_path),
            'file_identifier': str(file_identifier),
            'new_sha256': str(new_sha256),
            'old_sha256': str(old_sha256),
            'metadata': simple_metadata,
            'timestamp': time.time()
        })
        download_paths[str(file_identifier)] = str(local_path)
        logger.warning(f"⚠ Modified object downloaded: {file_identifier}")
        logger.warning(f"  Expected SHA256: {old_sha256}")
        logger.warning(f"  Actual SHA256: {new_sha256}")
    except Exception as e:
        logger.error(f"Error in handle_modified_object for {file_identifier}: {e}")
        if errors_encountered is not None:
            errors_encountered.append({
                'type': 'modified_object_callback',
                'file_identifier': str(file_identifier),
                'error': str(e)
            })

def safe_handle_missing_object(
    file_identifier: str,
    sha256: str,
    metadata: Dict[Hashable, Any]
) -> None:
    """Called when a specified object cannot be found."""
    try:
        global missing_objects
        
        # Serialize metadata to ensure it's picklable
        simple_metadata = serialize_metadata(metadata)
        
        missing_objects.append({
            'file_identifier': str(file_identifier),
            'sha256': str(sha256),
            'metadata': simple_metadata,
            'timestamp': time.time()
        })
        logger.error(f"✗ Missing object: {file_identifier}")
        logger.debug(f"  Expected SHA256: {sha256}")
    except Exception as e:
        logger.error(f"Error in handle_missing_object for {file_identifier}: {e}")
        if errors_encountered is not None:
            errors_encountered.append({
                'type': 'missing_object_callback',
                'file_identifier': str(file_identifier),
                'error': str(e)
            })

def safe_handle_new_object(
    local_path: str,
    file_identifier: str,
    sha256: str,
    metadata: Dict[Hashable, Any]
) -> None:
    """Called when a new object is found (GitHub specific)."""
    try:
        global new_objects
        
        # Serialize metadata to ensure it's picklable
        simple_metadata = serialize_metadata(metadata)
        
        new_objects.append({
            'local_path': str(local_path),
            'file_identifier': str(file_identifier),
            'sha256': str(sha256),
            'metadata': simple_metadata,
            'timestamp': time.time()
        })
        logger.info(f"+ New object found: {file_identifier}")
        logger.debug(f"  Local path: {local_path}")
    except Exception as e:
        logger.error(f"Error in handle_new_object for {file_identifier}: {e}")
        if errors_encountered is not None:
            errors_encountered.append({
                'type': 'new_object_callback',
                'file_identifier': str(file_identifier),
                'error': str(e)
            })

def download_objaverse_xl_with_callbacks(
    download_dir: str = "./objaverse_xl",
    file_types: List[str] = ['glb'],
    max_objects: Optional[int] = None,
    save_repo_format: Optional[str] = None,
    temp_dir: str = None
) -> Dict:
    """
    Download Objaverse XL dataset using callback functions for robust error handling.
    
    Args:
        download_dir: Directory to download objects to
        file_types: List of file types to download (e.g., ['glb', 'obj', 'fbx'])
        max_objects: Maximum number of objects to download (None for all)
        save_repo_format: Format to save GitHub repos ("zip", "tar", "tar.gz", "files", or None)
        temp_dir: Custom temporary directory (will be created if needed)
    
    Returns:
        Dictionary with download results
    """
    # Initialize multiprocessing-safe data structures
    global manager, found_objects, modified_objects, missing_objects, new_objects, download_paths, errors_encountered
    
    manager = Manager()
    found_objects = manager.list()
    modified_objects = manager.list()
    missing_objects = manager.list()
    new_objects = manager.list()
    download_paths = manager.dict()
    errors_encountered = manager.list()
    
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

    logger.info(f"Starting download of {len(filtered)} objects to /home/benzshawelt/.objaverse/")
    logger.info(f"Using temporary directory: {temp_dir}")
    
    start_time = time.time()

    try:
        # Try downloading with multiprocessing
        logger.info("Attempting download with multiprocessing")
        result = oxl.download_objects(
            objects=filtered,
            handle_found_object=safe_handle_found_object,
            handle_modified_object=safe_handle_modified_object,
            handle_missing_object=safe_handle_missing_object,
            handle_new_object=safe_handle_new_object,
            save_repo_format=save_repo_format,
            processes=mp.cpu_count(),  # Use all available cores
        )
        
        logger.info("Download process completed")
        
    except TypeError as e:
        if "processes" in str(e):
            # If processes parameter not supported, try without it
            logger.warning("Processes parameter not supported, trying without it")
            try:
                result = oxl.download_objects(
                    objects=filtered,
                    handle_found_object=safe_handle_found_object,
                    handle_modified_object=safe_handle_modified_object,
                    handle_missing_object=safe_handle_missing_object,
                    handle_new_object=safe_handle_new_object,
                    save_repo_format=save_repo_format,
                )
                logger.info("Download process completed")
            except Exception as e2:
                logger.error(f"Download process failed in oxl {e2}")
                result = None
        else:
            logger.error(f"Download process failed due to TypeError: {e}")
            result = None
    except Exception as e:
        logger.error(f"Download process failed due to Exception: {e}")
        result = None
    
    # Final cleanup
    cleanup_temp_files(temp_dir)
    
    # Convert Manager objects to regular Python objects for final report
    found_objects_list = list(found_objects)
    modified_objects_list = list(modified_objects)
    missing_objects_list = list(missing_objects)
    new_objects_list = list(new_objects)
    download_paths_dict = dict(download_paths)
    errors_encountered_list = list(errors_encountered)
    
    # Calculate summary
    duration = time.time() - start_time
    total_requested = len(filtered)
    
    summary = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'duration_seconds': round(duration, 2),
        'total_requested': total_requested,
        'found_objects': len(found_objects_list),
        'modified_objects': len(modified_objects_list),
        'missing_objects': len(missing_objects_list),
        'new_objects': len(new_objects_list),
        'errors_encountered': len(errors_encountered_list),
        'success_rate': len(found_objects_list) / total_requested * 100 if total_requested > 0 else 0,
        'objects_per_second': len(found_objects_list) / duration if duration > 0 else 0
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
        'found_objects': found_objects_list,
        'modified_objects': modified_objects_list,
        'missing_objects': missing_objects_list,
        'new_objects': new_objects_list,
        'download_paths': download_paths_dict,
        'errors_encountered': errors_encountered_list
    }
    
    try:
        with open(report_file, 'w') as f:
            json.dump(detailed_report, f, indent=4, default=str)
        logger.info(f"Detailed download report saved to {report_file}")
    except Exception as e:
        logger.error(f"Failed to save report: {e}")
    
    # Log summary
    logger.info(f"Download Summary:")
    logger.info(f"  Total requested: {summary['total_requested']}")
    logger.info(f"  Successfully downloaded: {summary['found_objects']}")
    logger.info(f"  Modified objects: {summary['modified_objects']}")
    logger.info(f"  Missing objects: {summary['missing_objects']}")
    logger.info(f"  New objects found: {summary['new_objects']}")
    logger.info(f"  Errors encountered: {summary['errors_encountered']}")
    logger.info(f"  Success rate: {summary['success_rate']:.2f}%")
    logger.info(f"  Download speed: {summary['objects_per_second']:.2f} objects/sec")
    
    # Log any errors encountered
    if errors_encountered_list:
        logger.warning(f"Encountered {len(errors_encountered_list)} errors during download:")
        for error in errors_encountered_list[:10]:  # Show first 10 errors
            logger.warning(f"  - {error['type']}: {error['file_identifier']} - {error['error']}")
        if len(errors_encountered_list) > 10:
            logger.warning(f"  ... and {len(errors_encountered_list) - 10} more errors")
    
    return {
        'success': True,
        'summary': summary,
        'download_paths': download_paths_dict,
        'detailed_report': detailed_report
    }

def download_objaverse_regular(num_objects: int = 100000, temp_dir: str = None):
    """
    Download regular Objaverse dataset with custom temp directory.
    """
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
    num_objects: int = 10000,  # Reduced default from 1,000,000 to 10,000
    file_types: List[str] = ['glb'],
    save_repos: bool = True,
    temp_dir: str = None
):
    """
    Main download function with custom temporary directory support.
    
    Args:
        xl: Whether to download Objaverse XL dataset
        num_objects: Number of objects to download
        file_types: List of file types to download for XL dataset
        save_repos: Whether to save GitHub repositories (XL only)
        temp_dir: Custom temporary directory path
    """
    if xl:
        logger.info("Starting Objaverse XL download with callback-based error handling")
        save_repo_format = "files" if save_repos else None
        max_objects = num_objects if num_objects < 100000 else None
        
        results = download_objaverse_xl_with_callbacks(
            file_types=file_types,
            max_objects=max_objects,
            save_repo_format=save_repo_format,
            temp_dir=temp_dir
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
    
    parser = argparse.ArgumentParser(description="Download Objaverse dataset using callback-based error handling.")
    parser.add_argument(
        "--xl",
        action="store_true",
        default=True,
        help="Download the Objaverse XL dataset instead of the regular Objaverse dataset.",
    )
    parser.add_argument(
        "--num-objects",
        type=int,
        default=100000,  # Reduced default to be more reasonable
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
    
    args = parser.parse_args()
    
    try:
        success = download_objaverse(
            xl=args.xl,
            num_objects=args.num_objects,
            file_types=args.file_types,
            save_repos=False,
            temp_dir=args.temp_dir
        )
        exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("Download interrupted by user")
        exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        exit(1)