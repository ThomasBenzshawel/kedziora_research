conda create -n pointmlp python=3.8 -y
conda activate pointmlp
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install cycler einops h5py pyyaml scikit-learn scipy tqdm matplotlib
pip install pointnet2_ops_lib/.
