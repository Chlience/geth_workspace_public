mkdir /3rdparty

apt update
apt install -y sudo htop fish iproute2 vim python3.10 python3-pip g++-11 wget git libgoogle-glog-dev libgflags-dev cmake libzmq3-dev automake

# install libgtest-dev
apt install -y libgtest-dev
cd /usr/src/googletest
mkdir build && cd build
cmake ..
make -j
make install -j

# install openmpi
cd /3rdparty/
wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.1.tar.gz
tar zxvf openmpi-4.0.1.tar.gz
cd openmpi-4.0.1
./configure
make -j
make install -j

# install metis
cd /3rdparty/
wget https://karypis.github.io/glaros/files/sw/metis/metis-5.1.0.tar.gz
tar zxvf metis-5.1.0.tar.gz
cd metis-5.1.0
make config shared=1
make -j
make install -j

sudo groupadd -g 1019 wpb
sudo useradd -u 1017 -g 1019 -m -s /bin/bash wpb
echo "wpb ALL=(ALL) NOPASSWD: ALL" | sudo tee /etc/sudoers.d/wpb
sudo chmod 440 /etc/sudoers.d/wpb