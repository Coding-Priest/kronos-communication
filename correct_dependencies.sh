# 1. Create the Conda environment with Python 3.8
conda create -n acsgd python=3.8

# 2. Activate the newly created environment
conda activate acsgd

# 3. Install PyTorch and TorchText for CUDA 11.1 (as specified by the project)
pip install torch==1.9.0+cu111 torchtext -f https://download.pytorch.org/whl/torch_stable.html

# 4. Install the CUDA 11.0 Toolkit *within the Conda environment*
#    (This provides the necessary libraries for cupy-cuda110)
conda install cudatoolkit=11.0 -c conda-forge

# 5. Install the specific CuPy version required by the project (for CUDA 11.0)
pip install cupy-cuda110==8.6.0

# 6. Install a version of NumPy compatible with cupy==8.6.0
#    (Must be < 1.24)
pip install "numpy<1.24"

# 7. Install the other specified dependencies
pip install datasets==2.2.2
pip install transformers==4.19.2
pip install sentencepiece==0.1.96

# 8. Optional: Verify installations (shows key packages)
conda list | grep -E 'torch|cupy|cudatoolkit|numpy|datasets|transformers|sentencepiece'
pip freeze | grep -E 'torch|cupy|numpy|datasets|transformers|sentencepiece'

# 9. Ready to run the project (remember network config if needed)
# export GLOO_SOCKET_IFNAME=...
# export NCCL_SOCKET_IFNAME=...
# sh run_rank0.sh