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

    local_path = path.join(__file__, "..")
    dir_name = path.dirname(path.abspath(sys.argv[0]))
    service_config = read_config(
        "service", config_path=path.join(dir_name, "config.yaml")
    )
    port = str(service_config["port"])

    subprocess.run(
        [
            "bentoml",
            "serve",
            "--working-dir",
            local_path,
            "service:svc",
            "--reload",
            "--port=" + port,
        ]
    )


if __name__ == "__main__":
    main()
