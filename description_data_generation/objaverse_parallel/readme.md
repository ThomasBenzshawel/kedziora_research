# Description Generation
(Inside /objaverse_parallel/description_parallel)
Make sure to delete the checkpoint folder if it is actually a new run!!!
python3 coordinate_parallel_processing.py --action submit --workers 7

# Object Download
(Inside /objaverse_parallel/download_parallel)

sbatch submit_obja_download.sbatch 

# Image Rendering
python3 coordinate_render_jobs.py --action submit --num_gpus 8

# Voxel Generation
sbatch submit_voxelize.sbatch

