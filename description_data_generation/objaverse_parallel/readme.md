# Description Generation
(Inside /objaverse_parallel/description_parallel)
python3 coordinate_parallel_processing.py --action submit --workers 8

# Object Download
(Inside /objaverse_parallel/download_parallel)

sbatch submit_obja_download.sbatch 

# Image Rendering
python3 coordinate_render_jobs.py --action submit --num_gpus 8

