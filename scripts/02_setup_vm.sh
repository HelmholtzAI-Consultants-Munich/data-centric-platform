# Connect to the machine via ssh
# ssh -v -i ~/.ssh/id_rsa_project1 ubuntu@134.94.88.118
# ssh -v -i ~/.ssh/id_rsa_project2 ubuntu@134.94.88.74

# Updates
echo "Updating distro"
sudo apt update && sudo apt upgrade

# Install NVIDIA drivers
echo "Installing NVIDIA driver"
sudo apt install -y gcc make g++
sudo apt-get install linux-headers-$(uname -r)
curl -o /tmp/NVIDIA-Driver.latest.run https://hpsrepo.fz-juelich.de/jusuf/nvidia/NVIDIA-Driver.latest
chmod 755 /tmp/NVIDIA-Driver.latest.run
sudo mkdir /etc/nvidia
curl -o /tmp/gridd.conf https://hpsrepo.fz-juelich.de/jusuf/nvidia/gridd.conf
sudo mv /tmp/gridd.conf /etc/nvidia/gridd.conf
sudo /tmp/NVIDIA-Driver.latest.run --ui=none --no-questions --disable-nouveau
# sudo reboot # run if needed

# Add the following to ~/.bashrc
#export PATH=/usr/local/cuda-11.8/bin${PATH:+:${PATH}}
#export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib\
#                         ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
# alias labelapp="cd /home/ubuntu/active-learning-platform && python al_framework.py"

# Install miniconda
echo "Installing Miniconda"
wget -O /tmp/Miniconda3-latest-Linux-x86_64.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash /tmp/Miniconda3-latest-Linux-x86_64.sh

# Install napari
# conda env remove -n .venv
echo "Installing napari"
conda activate base && pip install napari[all]
sudo apt-get install -y libdbus-1-3 libxkbcommon-x11-0 libxcb-icccm4 \
    libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 \
    libxcb-xinerama0 libxcb-xinput0 libxcb-xfixes0

# Install xpra
echo "Installing xpra"
DISTRO=focal
sudo apt-get install apt-transport-https software-properties-common
sudo apt install ca-certificates
sudo wget -O "/usr/share/keyrings/xpra-2022.gpg" https://xpra.org/xpra-2022.gpg
sudo wget -O "/usr/share/keyrings/xpra-2018.gpg" https://xpra.org/xpra-2018.gpg
sudo wget -O "/etc/apt/sources.list.d/xpra.list" https://xpra.org/repos/$DISTRO/xpra.list
sudo apt-get update
sudo apt-get -y install xpra xterm

# Install pytorch and cellpose
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
pip install pyqtgraph PyQt5 numpy numba scipy natsort cellpose

# Install github CLI for cloning private repo
conda install -y gh --channel conda-forge

# Login to github using tokens
# gh auth login
# gh auth logout