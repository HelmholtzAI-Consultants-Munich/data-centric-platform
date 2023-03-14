FROM continuumio/miniconda3

COPY . .

RUN apt-get update 
RUN apt-get -y install build-essential libaec-dev libblosc-dev

#RUN conda init bash
# Make RUN commands use the new environment:
RUN conda create -n dcp-env python=3.9
RUN echo "conda activate dcp-env" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]
RUN conda install scikit-image
RUN pip install -r requirements.txt

# The command below will also work instead of RUN echo and SHELL:
#RUN conda run -n dcp-env python -m pip install -r requirements.txt
WORKDIR ./dcp_server

CMD ["conda", "run", "-n", "dcp-env", "bentoml", "serve", "service:svc", "--reload", "--port=7010"]