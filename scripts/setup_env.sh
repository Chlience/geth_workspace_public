git clone git@github.com:VOIDMalkuth/gccl.git
git clone git@github.com:VOIDMalkuth/ragdoll.git
cd ragdoll
git submodule update --init
cd ..

# docker pull nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
docker run -it -d --name wpb_geth --privileged --gpus all -v $PWD:/workspace/ --shm-size 64g nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 bash
docker exec -it wpb_geth bash /workspace/dgcl/setup_env_in_docker.sh

