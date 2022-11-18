# Download your Openstack RC file from https://jusuf-cloud.fz-juelich.de/dashboard/project/api_access/openrc/
# then execute and login using your JUDOOR credentials
source ~/Downloads/poc-helmholtz-2023-openrc.sh

# https://apps.fz-juelich.de/jsc/hps/jusuf/cloud/first_steps_cloud.html#accessing-vms
# Upload keypair using own public key ~/.ssh/id_rsa_project1
openstack keypair create --public-key ~/.ssh/id_rsa_project1 project1_key
openstack keypair create --public-key ~/.ssh/id_rsa_project2 project2_key

# Create a network
openstack network create --no-share my-projects-internal-network
openstack subnet create my-projects-internal-network-subnet --network my-projects-internal-network \
 --subnet-range 10.0.0.0/8

# Create a router
openstack router create my-projects-internal-network-to-internet
openstack router set my-projects-internal-network-to-internet --external-gateway dmz-jusuf-cloud
openstack router add subnet my-projects-internal-network-to-internet my-projects-internal-network-subnet

# Create security group
openstack security group create test-securitygroup
openstack security group rule create --protocol tcp --dst-port 22 test-securitygroup
openstack security group rule create --protocol icmp test-securitygroup

# Create Ubuntu instance
openstack server create --flavor gpu.l --boot-from-volume 256 --security-group test-securitygroup --key-name \
project1_key --network my-projects-internal-network --image ubuntu-20.04 gpu-vm-project1

openstack server create --flavor gpu.l --boot-from-volume 256 --security-group test-securitygroup --key-name \
project2_key --network my-projects-internal-network --image ubuntu-20.04 gpu-vm-project2

# Create floating IP
openstack floating ip create dmz-jusuf-cloud

# Associate floating IP to VM {gpu-vm: 134.94.88.118,  gpu-vm2:134.94.88.74}
openstack server add floating ip gpu-vm-project1 134.94.88.118
openstack server add floating ip gpu-vm-project2 134.94.88.74
