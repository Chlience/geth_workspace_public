## 基础环境

```
nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
sha256:8f9dd0d09d3ad3900357a1cf7f887888b5b74056636cd6ef03c160c3cd4b1d95

docker run -it -d --name wpb_gccl --privileged --gpus all -v $PWD:/workspace/ --network host --shm-size 64g nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 bash
```

```
docker run -it -d --name wpb_geth_new --privileged --gpus all -v $PWD:/workspace/ --network host --shm-size 64g wpb_geth_image:v0 bash
```

```bash
apt install python3.10 python3-pip g++-11 wget git libgoogle-glog-dev libgflags-dev cmake libzmq3-dev automake
```

## GCCL安装

### libgtest-dev

```bash
apt install libgtest-dev
cd /usr/src/googletest
mkdir build && cd build
cmake ..
make -j
make install
```
### OpenMPI

```bash
wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.1.tar.gz
tar zxvf openmpi-4.0.1.tar.gz
cd openmpi-4.0.1
./configure
make -j
make install
```

安装至`/usr/local/`
### metis

```bash
wget http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/metis-5.1.0.tar.gz
tar zxvf metis-5.1.0.tar.gz
cd metis-5.1.0
make config shared=1
make -j
make install
```

安装至`/usr/local/`
### json
```bash
mkdir -p nlohmann/nlohmann && cd nlohmann/nlohmann
wget https://github.com/nlohmann/json/releases/download/v3.7.3/json.hpp
```

### 构建

修改`test/CMakeLists.txt`

```cmake
find_package(GTest REQUIRED)

if(GTest_FOUND)
  message(STATUS "Enable testing")

  set(GTEST_LIBRARY ${GTEST_LIBRARIES})
  set(GTEST_MAIN_LIBRARY ${GTEST_MAIN_LIBRARIES})

  # ......
  
endif()
```

修改单元测试`/workspace/gccl/test/gpu/kernel/primitives_unittest_gpu.cu`，直接将`test_cuda_utils.h`和`test_cuda_utils.cu`里面的Copy128bGlobal函数粘贴进去。

修改scripts/build.sh，移除g++-5的定义

```bash
export GCCL_CUDA=1     # Build with CUDA
export GCCL_GPU_TEST=1 # Build GPU test
export GCCL_MPI_TEST=1 # Build MPI test
export METIS_HOME=/usr/local # METIS installation directory
export METIS_DLL=/usr/local/lib/libmetis.so # METIS dll
export MPI_HOME=/usr/local # MPI installation directory
export NLOHMANN_HOME=/workspace/nlohmann # Nlohmann json installation home
export GCCL_HOME=/opt/gccl # GCCL installation home
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/

mkdir build
cp scripts/build.sh build/
cd build
./build.sh
make -j
make test
make install
```

注：`make test`可能报错`AllTestsIngcclMPIUnitTests`，但是直接运行`./test/gccl_mpi_unit_tests`是没有问题的

## Ragdoll

环境：torch 2.2.2+cu118; dgl 2.1.0+cu118;

```
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
pip install dgl==2.2.1 -f https://data.dgl.ai/wheels/torch-2.3/cu118/repo.html
```

编译

```bash
export GCCL_CUDA=1     # Build with CUDA
export GCCL_GPU_TEST=1 # Build GPU test
export GCCL_MPI_TEST=1 # Build MPI test
export METIS_HOME=/usr/local # METIS installation directory
export METIS_DLL=/usr/local/lib/libmetis.so # METIS dll
export MPI_HOME=/usr/local # MPI installation directory
export NLOHMANN_HOME=/workspace/nlohmann # Nlohmann json installation home
export GCCL_HOME=/opt/gccl # GCCL installation home
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/

./build.sh
```

### 其他依赖

```bash
pip install pyyaml pydantic
```


```bash
export GCCL_CUDA=1     # Build with CUDA
export GCCL_GPU_TEST=1 # Build GPU test
export GCCL_MPI_TEST=1 # Build MPI test
export METIS_HOME=/usr/local # METIS installation directory
export METIS_DLL=/usr/local/lib/libmetis.so # METIS dll
export MPI_HOME=/usr/local # MPI installation directory
export NLOHMANN_HOME=/workspace/dgcl/nlohmann # Nlohmann json installation home
export GCCL_HOME=/opt/gccl # GCCL installation home
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/
```

