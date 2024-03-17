import subprocess
import os

from dcp_client.utils.utils import get_relative_path
from dcp_client.app import DataSync


class DataRSync(DataSync):
    """
    Class which uses rsync bash command to sync data between client and server
    """

    def __init__(
        self,
        user_name: str,
        host_name: str,
        server_repo_path: str,
    ) -> None:
        """Constructs all the necessary attributes for the CustomRunnable.

        :param user_name: the user name of the server - if "local", then it is assumed that local machine is used for the server
        :type: user_name: str
        :param host_name: the host name of the server - if "local", then it is assumed that local machine is used for the server
        :type: host_name: str
        :param server_repo_path: the server path where we wish to sync data - if None, then it is assumed that local machine is used for the server
        :type server_repo_path: str
        """
        self.user_name = user_name
        self.host_name = host_name
        self.server_repo_path = server_repo_path

    def first_sync(self, path: str) -> tuple:
        """
        During the first sync the folder structure should be created on the server

        :param path: Path to the local directory to synchronize.
        :type path: str
        :return: result message of subprocess
        :rtype: tuple
        """
        server = self.user_name + "@" + self.host_name + ":" + self.server_repo_path
        try:
            # Run the subprocess command
            result = subprocess.run(["rsync", "-azP", path, server], check=True)
            return ("Success", result.stdout)
        except subprocess.CalledProcessError as e:
            return ("Error", e)

    def sync(self, src: str, dst: str, path: str) -> tuple:
        """Syncs the data between the src and the dst. Both src and dst can be one of either
        'client' or 'server', whereas path is the local path we wish to sync

        :param src: A string specifying the source, from where the data will be sent to dst. Can be 'client' or 'server'.
        :type src: str
        :param dst: A string specifying the destination, where the data from src will be sent to. Can be 'client' or 'server'.
        :type dst: str
        :param path: Path to the directory we want to synchronize.
        :type path: str
        :return: result message of subprocess
        :rtype: tuple

        """
        path += "/"  # otherwise it doesn't go in the directory
        rel_path = get_relative_path(path)  # get last folder, i.e. uncurated, curated
        server_full_path = os.path.join(self.server_repo_path, rel_path)
        server_full_path += "/"
        server = self.user_name + "@" + self.host_name + ":" + server_full_path
        print("server is: ", server)

        if src == "server":
            src = server
            dst = path
        else:
            src = path
            dst = server
        try:
            # Run the subprocess command
            _ = subprocess.run(["rsync", "-r", "--delete", src, dst], check=True)
            return ("Success", server_full_path)
        except subprocess.CalledProcessError as e:
            return ("Error", e)


if __name__ == "__main__":
    ds = DataRSync()  # vm2
    # These combinations work for me:
    # ubuntu@jusuf-vm2:/path...
    # jusuf-vm2:/path...
    dst = "server"
    src = "client"
    # dst = 'client'
    # src = 'server'
    # path = "data/"
    path = (
        "/Users/christina.bukas/Documents/AI_projects/code/data-centric-platform/data"
    )
    ds.sync(src, dst, path)
