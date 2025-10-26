## Things that are adjacent to our aproach
* https://www.designboom.com/technology/onformative-trains-ai-sculpt-3d-models-cube-voxels-12-22-2022/
* https://www.reddit.com/r/VoxelGameDev/comments/1146bhj/aigenerated_voxelized_3d_models/
* https://iliad.ai/canvas
    * ![alt text](image.png)
    * Clearly not ready to 3D Print
* https://drububu.com/miscellaneous/voxelizer/?out=obj
    * Yes, voxelizing an object is essentially solved

* Cheaper Meshy Exists: https://www.sloyd.ai/about
* Meshy: https://www.meshy.ai/
    * Meshy essentially generates a 3D object and then voxelizes it
    * We want to generate a voxel object a-la 3D printer style, then smooth

* ## Seems Very Similar to our aproach
    * https://research.nvidia.com/labs/toronto-ai/xcube/
        * Published at the end of 2023: https://arxiv.org/abs/2312.03806
        * Code Exists: https://github.com/nv-tlabs/XCube
        * Nice Surface reconstruction: https://research.nvidia.com/labs/toronto-ai/NKSR/
            * Code: https://github.com/nv-tlabs/nksr
            

    * https://www.gamedeveloper.com/business/sega-teams-with-ai-startup-eques-on-voxel-generation-tool
    * Text to voxel exists with Meshy: https://www.youtube.com/watch?v=0pIXhOXE1W0


# Very good Resource
    - https://www.meshlab.net/

# FVDB is very good for a representing our 3D data, no documentation tho
    - https://research.nvidia.com/labs/prl/publication/williams2024fvdb/


# Very good for training, never seen this before, but now we don't have to deal with making a sharding enabled model!
    - https://lightning.ai/docs/pytorch/stable/

# Other useful models

    - https://github.com/threestudio-project/threestudio?tab=readme-ov-file#zero-1-to-3-
    


# OUR DATA SET
* https://objaverse.allenai.org/



# More data?
- Shape Net
https://changeit3d.github.io/



# Data Augmentation
https://medium.com/lightricks-tech-blog/using-visual-llms-in-large-scale-multi-label-image-classification-pipelines-8fadbb6a1b2c

Simple enough to code out and get some clean data augmentation going for free.

https://support.prodi.gy/t/recipe-for-batch-wise-labelling-using-both-a-local-model-and-an-llm/7248
https://prodi.gy/buy

Code "recipe" market for this is a 400 dollar buy in.

https://arxiv.org/abs/2407.07053

https://encord.com/blog/top-multimodal-annotation-tools/