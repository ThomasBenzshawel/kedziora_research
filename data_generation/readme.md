# How to create the image description data


conda config --set channel_priority flexible

Conda env create -f objaverse_download.yaml
conda activate objaverse_download
python Objaverse_download.py

conda env create -f camera_env.yml
conda activate camera_dev
python Final_cam.py

conda env create -f llava_environment.yml

git clone https://github.com/LLaVA-VL/LLaVA-NeXT
cd LLaVA-NeXT
conda create -n llava python=3.10 -y
conda activate llava
python -m pip install --upgrade pip  # Enable PEP 660 support.
python -m pip install -e ".[train]"

<!-- python -m pip install flash-attn --no-build-isolationcd  : not working just yet--> 

cd ../
python interleved.py