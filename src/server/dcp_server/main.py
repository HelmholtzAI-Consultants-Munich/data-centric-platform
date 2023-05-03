import subprocess
from os import path
#import sys

def main():
    '''entry point to bentoml
    '''
    # global config_path
    # args = sys.argv
    # if len(args) > 1:
    #     if path.exists(args[1]) and args[1].endswith('.cfg'):
    #         config_path = args[1]
    # else:
    #     config_path = 'config.cfg'

    local_path = path.join(__file__, '..')

    subprocess.run([
        "bentoml",
        "serve", 
        '--working-dir', 
        local_path,
        "service:svc",
        "--reload",
        "--port=7010",
    ])
    


if __name__ == "__main__":
    main()
