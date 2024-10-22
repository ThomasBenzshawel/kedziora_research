# *ƒ*(VDB) is a feature branch for the openvdb project

https://github.com/AcademySoftwareFoundation/openvdb/tree/feature/fvdb

## Building *f*VDB from Source
*f*VDB is a Python library implemented as a C++ Pytorch extension.

**(Optional) Install libMamba for a huge quality of life improvement when using Conda**
```
conda update -n base conda
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
```

### Conda Environment

Next, create the `fvdb` conda environment by running the following command from the root of this repository, and then grabbing a ☕:
```shell
conda env create -f env/test_environment.yml
```

**Note:**  You can optionally use the `env/build_environment.yml` environment file if you want a minimum set of dependencies needed to build *f*VDB and don't intend to run the tests or the `env/learn_environment` if you would like the additional packages needed to run the examples and view their visualizations.

Now activate the environment:
```shell
conda activate fvdb_test
```


### Building *f*VDB

**:warning: Note:** Compilation can be very memory-consuming. We recommend setting the `MAX_JOBS` environment variable to control compilation job parallelism with a value that allows for one job every 2.5GB of memory:

```bash
export MAX_JOBS=$(free -g | awk '/^Mem:/{jobs=int($4/2.5); if(jobs<1) jobs=1; print jobs}')
```

You shoulddo an editable install with setuptools:
```shell
pip install -e .
```
or directly install it to your site package folder if you are developing extensions:
```shell
pip install .
```

Afterwards, find your fvdb.so and add it to your python path and python lib like so:

# First, activate your environment if it's not already activated
conda activate fvdb_test

# Find the path to your environment
CONDA_ENV_PATH=$(conda info --base)/envs/fvdb_test

# Create the activation directory if it doesn't exist
mkdir -p $CONDA_ENV_PATH/etc/conda/activate.d

# Create a new script to set the environment variables
cat << EOF > $CONDA_ENV_PATH/etc/conda/activate.d/env_vars.sh
#!/bin/bash
export LD_LIBRARY_PATH="$CONDA_ENV_PATH/lib:$CONDA_ENV_PATH/lib/python3.10/site-packages/fvdb-0.0.1-py3.10-linux-x86_64.egg/fvdb:\$LD_LIBRARY_PATH"
export PYTHONPATH="$CONDA_ENV_PATH/lib/python3.10/site-packages/fvdb-0.0.1-py3.10-linux-x86_64.egg:\$PYTHONPATH"
EOF

# Make the script executable
chmod +x $CONDA_ENV_PATH/etc/conda/activate.d/env_vars.sh

# Create a deactivation script to unset the variables when the environment is deactivated
mkdir -p $CONDA_ENV_PATH/etc/conda/deactivate.d
cat << EOF > $CONDA_ENV_PATH/etc/conda/deactivate.d/env_vars.sh
#!/bin/bash
unset LD_LIBRARY_PATH
unset PYTHONPATH
EOF

# Make the deactivation script executable
chmod +x $CONDA_ENV_PATH/etc/conda/deactivate.d/env_vars.sh

# Deactivate and reactivate the environment to apply changes
conda deactivate
conda activate fvdb_test

# Verify the changes
echo $LD_LIBRARY_PATH
echo $PYTHONPATH



### Running Tests

To make sure that everything works by running tests:
```shell
pytest tests/unit
```

### Building Documentation

To build the documentation, simply run:
```shell
python setup.py build_ext --inplace
sphinx-build -E -a docs/ build/sphinx
# View the docs
open build/sphinx/index.html
```

### Docker Image

To build and test *f*VDB, we have the dockerfile available:
```shell
# Build fvdb
docker build . -t fvdb-dev
# Run fvdb (or replace with your command)
docker run -it --gpus all --rm \
  --user $(id -u):$(id -g) \
  --mount type=bind,source="$HOME/.ssh",target=/root/.ssh \
  --mount type=bind,source="$(pwd)",target=/fvdb \
  fvdb-dev:latest \
  conda run -n fvdb_test --no-capture-output python setup.py develop
```
