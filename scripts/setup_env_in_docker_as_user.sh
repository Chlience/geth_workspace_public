cd /workspace

# json
# source /workspace/env_vars.sh
# rm -rf nlohmann/nlohmann && mkdir -p nlohmann/nlohmann && cd nlohmann/nlohmann
# wget https://github.com/nlohmann/json/releases/download/v3.7.3/json.hpp

# instal gccl
source /workspace/env_vars.sh
cd /workspace/gccl
rm -rf build && mkdir -p build
cp scripts/build.sh build/
cd build
./build.sh
make -j
# make test
sudo mkdir -p /opt/gccl
sudo chown -R wpb:wpb /opt/gccl
make install -j

# install ragdoll
source /workspace/env_vars.sh
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
pip install dgl==2.2.1 -f https://data.dgl.ai/wheels/torch-2.3/cu118/repo.html
pip install pyyaml pydantic
pip install torchdata==0.9.0
pip install ogb

cd /workspace/ragdoll
rm -rf build
rm -rf build/lib.linux-x86_64-3.10/ragdoll/ragdoll_core.cpython-310-x86_64-linux-gnu.so
./build.sh
rm -rf build/lib.linux-x86_64-3.10/ragdoll/ragdoll_core.cpython-310-x86_64-linux-gnu.so
pip install -e .
