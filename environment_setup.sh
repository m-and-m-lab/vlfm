#!/bin/bash
# Script that sets up any pip installs that need the --no-deps flag.
# Installs vlfm into the environment.

# Set CUDA_HOME
export CUDA_HOME=$CONDA_PREFIX

# install requirements
pip install -r requirements.txt

# Set cmake prefix path for installing sophus
SITE_PACKAGES_DIR=$(python -c "import site; print(site.getsitepackages()[0])")
PYBIND_11_DIR="$SITE_PACKAGES_DIR/pybind11/share/cmake/pybind11"

# Check if the directory exists
if [ ! -d "$PYBIND_11_DIR" ]; then
    echo "Failed to locate pybind11 directory for cmake prefix path, which is used for installing sophus."
    exit 1
fi

# Set the cmake prefix path
export CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:$PYBIND_11_DIR

pip install --no-deps webdataset===0.1.40 faster-fifo open3d==0.19.0 sophuspy===0.0.8 decord\
        git+https://github.com/naokiyokoyama/frontier_exploration.git\
        git+https://github.com/ChaoningZhang/MobileSAM.git\
        git+https://github.com/naokiyokoyama/depth_camera_filtering\
        git+https://github.com/naokiyokoyama/bd_spot_wrapper.git

# Remove grounding dino if it exists since we need to pull from github
rm -rf GroundingDINO

# Clone GroundingDINO
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip install -e .

# Back to vlfm root
cd ..

# Clone yolov7
rm -rf yolov7
git clone git@github.com:WongKinYiu/yolov7.git

# weights for grounding DINO and other models
mkdir -p GroundingDINO/weights
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth -O GroundingDINO/weights/groundingdino_swint_ogc.pth
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt -O data/yolov7-e6e.pt
wget https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt -O data/mobile_sam.pt
cp GroundingDINO/weights/groundingdino_swint_ogc.pth data/

# install vlfm
pip install -e .[reality]

# TODO: Remove this step once saleforce-lavis is replaced with transformers library calls
# Install older version of timm to ensure VLFM runs
pip install timm==0.6.12