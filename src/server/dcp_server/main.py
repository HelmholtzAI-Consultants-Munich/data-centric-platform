import subprocess
from os import path


def main(args=None):
    '''entry point to bentoml
    '''
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