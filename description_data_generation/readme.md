# How to create the image description data


conda config --set channel_priority flexible

Conda env create -f objaverse_download.yaml
conda activate objaverse_download
python objaverse_download.py

conda env create -f camera_env.yml
conda activate camera
python final_cam.py

You wll get errors for 2-channel textures, those objects will be ignored, they were luminesence based objects
There will also be warnings for black and white images, those are typically objects that are not within the veiw of the camera (either they are too large, or were never put in the .glb file)
There are a few objects that have multiple files within it for one reason or another, those will give you the Expected a Trimesh or a list, got a <class 'trimesh.path.path.Path3D'> error.

<!-- conda env create -f llava_environment.yml -->

git clone https://github.com/LLaVA-VL/LLaVA-NeXT
cd LLaVA-NeXT
conda create -n llava python=3.10 -y
conda activate llava
python -m pip install --upgrade pip  # Enable PEP 660 support.
python -m pip install -e ".[train]"

Optionally, install
python -m pip install flash-attn --no-build-isolation

cd ../
python interleved.py # For simpler description
OR
pip install objaverse
python COT_interleaved.py

If you would like to skip the full model / prompt evaluation pipeline, you only need to evaluate *12 items* to be confident in your quality of your pipeline

add to a .env file in the same dir for your cloudinary service:
MONGO_URI=...
DATABASE=...
API_BASE_URL=...

conda env create -f cloudinary.yaml
conda activate cloudinary
python upload_to_mongo.py --dir objaverse_descriptions
python upload_to_cloudinary.py --dir objaverse_images

