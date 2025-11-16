#!/usr/bin/env python3
"""
Preprocessing script to scan directories and create a cached file list.
This runs once before all workers start, avoiding redundant directory scanning.
"""

import json
import argparse
import logging
from pathlib import Path
import sys
from datetime import datetime

def setup_logging():
    """Set up logging configuration"""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)
def scan_directories(scan_dirs, logger):
    """Scan directories with deduplication"""
    seen_paths = set()  # Track absolute paths we've already added
    items = []
    
    for scan_dir in scan_dirs:
        if scan_dir is None:
            continue
            
        base_path = Path(scan_dir)
        if not base_path.exists():
            logger.warning(f"Scan directory does not exist: {base_path}")
            continue
        
        logger.info(f"Scanning directory for GLB files: {base_path}")
        start_count = len(items)
        duplicates = 0
        
        for glb_path in base_path.rglob('*.glb'):
            try:
                abs_path_str = str(glb_path.resolve())  # resolve() follows symlinks
                
                # Skip if we've already seen this exact path
                if abs_path_str in seen_paths:
                    duplicates += 1
                    continue
                
                seen_paths.add(abs_path_str)
                uid = glb_path.stem
                items.append((uid, abs_path_str))
                
            except Exception as e:
                logger.error(f"Failed to process path {glb_path}: {e}")
                continue
        
        found_in_dir = len(items) - start_count
        logger.info(f"Found {found_in_dir} GLB files in directory: {base_path}")
        logger.info(f"Skipped {duplicates} duplicate paths (symlinks/overlaps)")
    
    logger.info(f"Total unique GLB files: {len(items)}")
    return items

    
def main():
    parser = argparse.ArgumentParser(
        description='Scan directories for GLB files and create cached file list'
    )
    
    parser.add_argument('--scan_dir', required=True, 
                        help='Primary directory to scan for GLB files')
    parser.add_argument('--scan_dir_2', required=False, 
                        help='Secondary directory to scan for GLB files')
    parser.add_argument('--output_file', required=True,
                        help='Output JSON file to store the file list')
    
    args = parser.parse_args()
    logger = setup_logging()
    
    logger.info("="*60)
    logger.info("GLB File List Preprocessing")
    logger.info(f"Started at: {datetime.now()}")
    logger.info("="*60)
    
    # Collect scan directories
    scan_dirs = [args.scan_dir]
    if args.scan_dir_2:
        scan_dirs.append(args.scan_dir_2)
    
    logger.info(f"Directories to scan: {scan_dirs}")
    
    try:
        # Scan directories
        items = scan_directories(scan_dirs, logger)
        
        # Prepare output
        output_data = {
            'scan_timestamp': datetime.now().isoformat(),
            'scan_directories': scan_dirs,
            'total_files': len(items),
            'files': items  # List of [uid, path] pairs
        }
        
        # Save to JSON
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"File list saved to: {output_path}")
        logger.info(f"Total files: {len(items)}")
        logger.info("="*60)
        logger.info("Preprocessing completed successfully!")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()