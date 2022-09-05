FROM 812206152185.dkr.ecr.us-west-2.amazonaws.com/latch-base:6839-main

###### Get conda and download dependencies into it
RUN apt-get install -y curl unzip
# Activate conda, install all dependencies for later use in an environment, then switch back to base
RUN curl -L -O \
    https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Mambaforge-Linux-x86_64.sh -b \
    && rm -f Mambaforge-Linux-x86_64.sh

# Create the environment
ENV PATH /root/mambaforge/bin:$PATH
COPY task_environment.yml /root/
RUN . /root/mambaforge/etc/profile.d/conda.sh &&\
    . /root/mambaforge/etc/profile.d/mamba.sh &&\
    mamba activate base &&\
    mamba env create -f task_environment.yml 

# Download local dependencies, include latch so that it caches
COPY requirements.txt /root/
RUN python3 -m pip install -r requirements.txt
######

# Copy scripts, data, and other files to the container
######
COPY scripts /root/scripts/
COPY data /root/data/
COPY unirep_source /root/unirep_source/
RUN chmod +x /root/scripts/*.py
######

RUN echo 'installing modified latch '
# RUN python3 -m pip install --upgrade /root/data/latch-1.15.0-py3-none-any.whl

# Required commands
######
COPY wf /root/wf/
ARG tag
ENV FLYTE_INTERNAL_IMAGE $tag
RUN sed -i 's/latch/wf/g' flytekit.config
RUN python3 -m pip install --upgrade latch
WORKDIR /root
######