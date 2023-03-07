FROM 812206152185.dkr.ecr.us-west-2.amazonaws.com/latch-base:ace9-main
run pip install latch==2.13.1
run mkdir /opt/latch

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

# Required commands
######
# copy all code from package (use .dockerignore to skip files)
COPY . /root/

# Enable scripts
######
RUN chmod +x /root/scripts/*.py
RUN chmod +x /root/test_scripts/*.py

# latch internal tagging system + expected root directory --- changing these lines will break the workflow
ARG tag
ENV FLYTE_INTERNAL_IMAGE $tag
WORKDIR /root
