FROM 812206152185.dkr.ecr.us-west-2.amazonaws.com/latch-base:02ab-main

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
COPY environment.yml /root/environment.yml
RUN . /root/mambaforge/etc/profile.d/conda.sh &&\
    . /root/mambaforge/etc/profile.d/mamba.sh &&\
    mamba activate base &&\
    mamba env create -f environment.yml 
    # mamba activate unirep
RUN python3 -m pip install biopython requests
RUN python3 -m pip install latch
######


#####
# permanently run conda activate 
# puts your environment first in places to look
# ENV PATH /root/mambaforge/envs/nf-core-sarek-2.7.1/bin:$PATH
ENV PATH $PATH:/root/mambaforge/envs/unirep/bin
# ENV PATH /root/mambaforge/etc/profile.d:$PATH
# ENV PATH /root/mambaforge/bin:$PATH
# COPY /root/mambaforge/* .
#####


######
# RUN mkdir /root/wf
COPY . .
RUN chmod +x /root/scripts/*.py
RUN chmod +x /root/tutorial.py /root/unirep.py /root/data_utils.py
######


######
ARG tag
ENV FLYTE_INTERNAL_IMAGE $tag
RUN sed -i 's/latch/wf/g' flytekit.config
RUN python3 -m pip install --upgrade latch
WORKDIR /root
######