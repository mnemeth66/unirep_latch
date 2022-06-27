FROM 812206152185.dkr.ecr.us-west-2.amazonaws.com/latch-base:02ab-main

RUN apt-get update && apt-get install -y python3-pip
RUN pip freeze > requirements.txt
RUN python3 -m pip install -r requirements.txt

RUN curl -L -O \
    https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Mambaforge-Linux-x86_64.sh -b \
    && rm -f Mambaforge-Linux-x86_64.sh

RUN /root/mambaforge/bin/conda config --set channel_priority strict


# Now we build our environment so that all downloaded packages are accessible
# to the workflow at execution time.
#RUN /root/mambaforge/bin/mamba create -n tensorflow \
#  && /root/mambaforge/bin/mamba clean -a

RUN /root/mambaforge/bin/mamba install libmambapy=0.16 -c conda-forge
RUN /root/mambaforge/bin/mamba install python=3.6 -c conda-forge
RUN /root/mambaforge/bin/mamba install tensorflow=1.3.0 -c conda-forge


COPY data /root/data
COPY tutorial.py /root/tutorial.py
COPY unirep.py /root/unirep.py
COPY data_utils.py /root/data_utils.py

COPY wf /root/wf

ARG tag
ENV FLYTE_INTERNAL_IMAGE $tag
RUN  sed -i 's/latch/wf/g' flytekit.config
RUN python3 -m pip install --upgrade latch
WORKDIR /root
