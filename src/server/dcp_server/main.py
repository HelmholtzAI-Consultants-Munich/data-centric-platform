import subprocess

def main(args=None):
    '''entry point to bentoml
    '''
    subprocess.run(["bentoml","serve","service:svc","--reload","--port=7010"])
    
if __name__ == "__main__":
    main()