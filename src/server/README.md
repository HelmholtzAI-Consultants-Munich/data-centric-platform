# data-centric-platform Server

The client and server communicate via the [bentoml](https://www.bentoml.com/?gclid=Cj0KCQiApKagBhC1ARIsAFc7Mc6iqOLi2OcLtqMbGx1KrFjtLUEZ-bhnqlT2zWREE0x7JImhtNmKlFEaAvSSEALw_wcB) library. The client interacts with the server every time we run model inference or training, so the server should be running before starting the client.

Once the server is running, you can verify it is working by visiting http://localhost:7010/ in your web browser.s

## Platforms

### Mac with Anaconda

Installation:
```
conda env create -f environment_dcp.yml
```

Run:
```
conda run -n dcp-env bentoml serve service:svc --reload --port=7010
```


### Docker-Compose
```
docker compose up
```

### Docker Non-Interactively
```
docker build -t dcp-server .
docker run -p 7010:7010 -it dcp-server
```

### Docker Interactively
```
docker build -t dcp-server .
docker run -it dcp-server bash
bentoml serve service:svc --reload --port=7010
```
