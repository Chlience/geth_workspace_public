export PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/lib/"

export GCCL_CUDA=1     # Build with CUDA
export GCCL_GPU_TEST=1 # Build GPU test
export GCCL_MPI_TEST=1 # Build MPI test
export METIS_HOME=/usr/local # METIS installation directory
export METIS_DLL=/usr/local/lib/libmetis.so # METIS dll
export MPI_HOME=/usr/local # MPI installation directory
export NLOHMANN_HOME=/workspace/nlohmann # Nlohmann json installation home
export GCCL_HOME=/opt/gccl # GCCL installation home
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/:$HOME/.local/bin/

export PATH="$PATH:$HOME/.local/bin/"

export GCCL_CONFIG=/workspace/gccl/configs/gpu8.json

ulimit -c 0

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

