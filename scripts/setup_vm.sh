# Connect to machine via ssh
# ssh -v -i ~/.ssh/grey_haicu ubuntu@134.94.88.118

# Updates
sudo apt update && sudo apt upgrade

# Install NVIDIA drivers
sudo apt install -y gcc make g++
sudo apt-get install linux-headers-$(uname -r)
curl -o /tmp/NVIDIA-Driver.latest.run https://hpsrepo.fz-juelich.de/jusuf/nvidia/NVIDIA-Driver.latest
chmod 755 /tmp/NVIDIA-Driver.latest.run
sudo mkdir /etc/nvidia
curl -o /tmp/gridd.conf https://hpsrepo.fz-juelich.de/jusuf/nvidia/gridd.conf
sudo mv /tmp/gridd.conf /etc/nvidia/gridd.conf
sudo /tmp/NVIDIA-Driver.latest.run --ui=none --no-questions --disable-nouveau
# sudo reboot # run if needed