#!/bin/bash

MOD=langchain

SIF=${MOD}.sif

SINGULARITY=apptainer

${SINGULARITY} shell --no-home \
               --bind /home/kanazawa/models:/opt/models \
               --bind $(pwd):/opt/work \
               --pwd /opt/work \
               --shell /bin/bash \
               ${SIF}
