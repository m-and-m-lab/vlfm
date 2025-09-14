mamba create -y -n vlfm python=3.9
mamba activate vlfm

# compile dependencies
mamba install -y -c pytorch -c nvidia pytorch=1.13.1 torchvision=0.14.1 torchaudio=0.13.1 pytorch-cuda=11.7 mkl=2024.0.0 fairscale
mamba install -y -c nvidia cuda=11.7 cuda-nvcc=11.7 cuda-cccl=11.7
mamba install -y gxx=8.5 cmake pybind11 python-devtools make git git-lfs

# runtime dependencies
mamba install -y -c pyg pyg
mamba install -y -c aihabitat withbullet habitat-sim=0.2.5 headless
pip install --no-deps -e habitat-lab/habitat-lab  # install habitat_lab
pip install --no-deps -e habitat-lab/habitat-baselines
mamba install -y -c fvcore fvcore
mamba install -y pandas hydra-core gym=0.22.0 opencv trimesh=3 tensorboard ifcfg python-lmdb simplejson threadpoolctl pybullet\
        plotly dash addict scikit-learn scikit-image scikit-fmm pinocchio loguru yacs openai-clip fvcore iopath pycocotools\
        timm transformers einops spacy pycocoevalcap braceexpand gdown natsort pytorch_geometric supervision yapf nvtx
pip install bosdyn-client==4.0.3 bosdyn-api==4.0.3 six==1.16.0
pip install --no-deps webdataset===0.1.40 faster-fifo open3d sophuspy===0.0.8 salesforce-lavis decord\
        git+https://github.com/naokiyokoyama/frontier_exploration.git\
        git+https://github.com/ChaoningZhang/MobileSAM.git\
        git+https://github.com/naokiyokoyama/depth_camera_filtering\
        git+https://github.com/naokiyokoyama/bd_spot_wrapper.git\
        git+https://github.com/IDEA-Research/GroundingDINO.git\
        
# weights for grounding DINO
mkdir GroundingDINO/weights
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth -O GroundingDINO/weights/groundingdino_swint_ogc.pth

wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt -O data/yolov7-e6e.pt
wget https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt -O data/mobile_sam.pt
wget https://github.com/bdaiinstitute/vlfm/raw/main/data/pointnav_weights.pth -O data/pointnav_weights.pt
cp GroundingDINO/weights/groundingdino_swint_ogc.pth data/
# install vlfm
pip install --no-deps -e .[reality]