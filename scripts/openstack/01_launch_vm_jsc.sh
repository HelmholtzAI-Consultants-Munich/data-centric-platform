# Download your Openstack RC file from https://cloud.jsc.fz-juelich.de/project/api_access/ 
# then execute and login using your JUDOOR credentials
source ~/Downloads/poc-helmholtz-2023-openrc.sh

# https://apps.fz-juelich.de/jsc/hps/jsccloud/access_cloud.html
# Upload keypair using own public key ~/.ssh/id_rsa
openstack keypair create --public-key ~/.ssh/id_rsa project_key

# Create a network
openstack network create --no-share my-projects-internal-network
openstack subnet create my-projects-internal-network-subnet --network my-projects-internal-network \
 --subnet-range 192.0.2.0/24

# Create a router
openstack router create my-projects-internal-network-to-internet
openstack router set my-projects-internal-network-to-internet --external-gateway dmz-jusuf-cloud
openstack router add subnet my-projects-internal-network-to-internet my-projects-internal-network-subnet

# Create security group
openstack security group create test-securitygroup
openstack security group rule create --protocol tcp --dst-port 22 test-securitygroup
openstack security group rule create --protocol icmp test-securitygroup

# Create VM 
openstack server create --flavor SCS-16L:64:20n-z2-GNv:80 --security-group test-securitygroup \
    --key-name bukas1 --network my-projects-internal-network --image "RockyLinux 9.3" gpu-vm

# Create floating IP
# openstack floating ip create dmz-jusuf-cloud
# Associate floating IP to VM {gpu-vm: 134.94.198.230        }
# openstack server add floating ip gpu-vm 134.94.198.230

