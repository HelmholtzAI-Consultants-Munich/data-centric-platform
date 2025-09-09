from os import path
import sys
import subprocess

from dcp_server.utils.helpers import read_config


def main() -> None:
    """
    Contains main functionality related to the server.
    """
    # global config_path
    # args = sys.argv
    # if len(args) > 1:
    #     if path.exists(args[1]) and args[1].endswith('.cfg'):
    #         config_path = args[1]
    # else:
    #     config_path = 'config.cfg'


    local_path = path.join(path.dirname(path.realpath(__file__)))
    
    dir_name = path.dirname(path.abspath(sys.argv[0]))
    service_config = read_config(
        "service", config_path=path.join(dir_name, "config.yaml")
    )
    service_name = str(service_config["service_name"])
    port = str(service_config["port"])
    timeout = str(service_config["timeout"])
    
    subprocess.run(
        [
            "bentoml",
            "serve",
            "service:"+service_name,
            "--working-dir",
            local_path,
            "--reload",
            "--port=" + port,
            "--timeout=" + timeout
        ]
    ) 


if __name__ == "__main__":
    main()
