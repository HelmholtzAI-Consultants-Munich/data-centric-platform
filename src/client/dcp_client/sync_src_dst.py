import subprocess
from dcp_client.utils import get_relative_path
from dcp_client.app import DataSync

class DataRSync(DataSync):
    '''
    Class which uses rsync bash command to sync data between client and server
    '''
    def __init__(self,
                 host_name=None,
                 host_ip=None,
    ):
        """Constructs all the necessary attributes for the CustomRunnable.

        :param host_name: the host name of the server - if None, then it is assumed that local machine is used for the server
        :type: host_name: str
        :param host_ip: the host ip of the server - if None, then it is assumed that local machine is used for the server
        :type host_ip: str
        """  
        self.host_name = host_name
        self.host_ip = host_ip

    def sync(self, src, dst, path):
        rel_path = get_relative_path(path)
        server = self.host_name+"@"+self.host_ip+":"+rel_path+"/"
        if src=='server':
            src = server
            dst = path
        else:
            src = path
            dst = server
        
        subprocess.run(["rsync",
                        "-r" ,
                        "--delete", 
                        src, 
                        dst])
        return rel_path
        

if __name__=="__main__":
    ds = DataRSync(host_name="ubuntu",
                   host_ip="134.94.88.74")
    dst = "server"
    src = "client"
    path = "data/"
    ds.sync(src, dst, path)