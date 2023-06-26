import subprocess
from dcp_client.utils import get_relative_path
from dcp_client.app import DataSync

class DataRSync(DataSync):
    '''
    Class which uses rsync bash command to sync data between client and server
    '''
    def __init__(self,
                 user_name=None,
                 host_name=None,
                 server_repo_path='/home/ubuntu/data-centric-platform/data-sync/'
    ):
        """Constructs all the necessary attributes for the CustomRunnable.

        :param host_name: the host name of the server - if None, then it is assumed that local machine is used for the server
        :type: host_name: str
        :param host_ip: the host ip of the server - if None, then it is assumed that local machine is used for the server
        :type host_ip: str
        """  
        self.user_name = user_name
        self.host_name = host_name
        self.server_repo_path = server_repo_path

        self.server = self.user_name + "@" + self.host_name + ":" + self.server_repo_path

    def sync(self, src, dst, path):

        print(f'server is {self.server}')
        if src=='server':
            src = self.server
            dst = path
        else:
            src = path
            dst = self.server
        
        subprocess.run(["rsync",
                        "-r" ,
                        "--delete", 
                        src, 
                        dst])
        #return rel_path
        return self.server_repo_path
        

if __name__=="__main__":
    ds = DataRSync(user_name="ubuntu",
                   host_name="jusuf-vm2", 
                   server_repo_path="/home/ubuntu/data-centric-platform/data-sync/") #vm2
    # These combinations work for me:
    # ubuntu@jusuf-vm2:/path...
    # jusuf-vm2:/path...
    dst = "server"
    src = "client"
    # dst = 'client'
    # src = 'server'
    #path = "data/"
    path = "/Users/helena.pelin/Desktop/images/test/"
    ds.sync(src, dst, path)