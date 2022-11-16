# https://apps.fz-juelich.de/jsc/hps/jusuf/cloud/first_steps_cloud.html#accessing-vms
# Upload keypair using own public key ~/.ssh/grey_haicu.pub
openstack keypair create --public-key ~/.ssh/grey_haicu.pub my_user

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

# Create Ubuntu instance with gpu.xl flavor
openstack server create --flavor gpu.xl --security-group test-securitygroup --key-name my_user \
--network my-projects-internal-network --image ubuntu-20.04 gpu-vm

# Create floating IP using GUI for now

# Associate floating IP to VM
openstack server add floating ip gpu-vm 134.94.88.118

# Create persistent volume
openstack volume create --size 256 vol1
openstack volume create --size 256 vol2