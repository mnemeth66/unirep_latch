FROM 812206152185.dkr.ecr.us-west-2.amazonaws.com/latch-base:02ab-main

# Activate conda, install all dependencies for later use in an environment, then switch back to base
RUN curl -L -O \
    https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Mambaforge-Linux-x86_64.sh -b \
    && rm -f Mambaforge-Linux-x86_64.sh

# Create the environment
ENV PATH /root/mambaforge/bin:$PATH
COPY environment.yml /root/environment.yml
RUN . /root/mambaforge/etc/profile.d/conda.sh &&\
    . /root/mambaforge/etc/profile.d/mamba.sh &&\
    mamba activate base &&\
    mamba env create -f environment.yml


COPY data /root/data
COPY tutorial.py /root/tutorial.py
COPY unirep.py /root/unirep.py
COPY data_utils.py /root/data_utils.py
RUN chmod +x /root/tutorial.py /root/unirep.py /root/data_utils.py

COPY wf /root/wf

ARG tag
ENV FLYTE_INTERNAL_IMAGE $tag
RUN sed -i 's/latch/wf/g' flytekit.config
RUN python3 -m pip install --upgrade latch
WORKDIR /root
