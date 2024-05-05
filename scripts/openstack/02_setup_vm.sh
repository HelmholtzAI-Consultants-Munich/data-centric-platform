# Connect to the machine via ssh
# ssh -v -i ~/.ssh/id_rsa rocky@134.94.198.230 # IP should reflect the created floatingIP

# Bring VM to latest state and install some dependencies
sudo dnf update -y # cosmetic for getting the latest kernel in
sudo shutdown 1 -r
sudo dnf config-manager --set-enabled crb
sudo dnf -y install epel-release
sudo dnf install -y gcc gcc-c++ make kernel-headers-$(uname -r) kernel-devel-$(uname -r) tar bzip2 automake elfutils-libelf-devel libglvnd libglvnd-devel libglvnd-opengl libglvnd-glx acpid pciutils dkms

# Install Nvidia driver
sudo dnf config-manager --add-repo http://developer.download.nvidia.com/compute/cuda/repos/rhel9/$(uname -i)/cuda-rhel9.repo
sudo dnf module -y install nvidia-driver:latest-dkms
sudo shutdown 1 -r

# After rebooting the device, check the driver is instlled correctly
nvidia-smi

# Install miniconda
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Login to github using tokens
# gh auth login
# gh auth logout