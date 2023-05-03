import subprocess
from pathlib import PurePath
from dcp_client.app import DataSync

class DataRSync(DataSync):
    def __init__(self,
                 ssh_key,
                 host_name,
                 host_ip,
    ):
        self.ssh_key = ssh_key
        self.host_name = host_name
        self.host_ip = host_ip

    def sync(self, src, dst, path):
        server = self.host_name+"@"+self.host_ip+":"+PurePath(path).name+"/"
        if src=='server':
            src = server
            dst = path
        else:
            src = path
            dst = server
        subprocess.run(["rsync",
                        "-r" ,
                        "--delete", 
                        "-e",
                        "ssh -i "+self.ssh_key, 
                        src, 
                        dst])

if __name__=="__main__":
    ds = DataRSync(ssh_key="/Users/christina.bukas/.ssh/id_rsa_project2",
                   host_name="ubuntu",
                   host_ip="134.94.88.74")
    dst = "client"
    src = "server"
    path = "data/"
    ds.sync(src, dst, path)