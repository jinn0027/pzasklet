#!/bin/bash

MOD=langchain

SIF=${MOD}.sif

SINGULARITY=apptainer

#MODEL_DIR=/home/kanazawa/models
MODEL_DIR=/mnt/pool3/scratch/ishikawa/models

${SINGULARITY} -q exec --no-home \
               --bind ${MODEL_DIR}:/opt/models \
               --bind $(pwd):/opt/work \
               --pwd /opt/work \
               --writable-tmpfs \
               ${SIF} bash -c \
               "export HF_HOME=/opt/work/hf_cache && pip uninstall -y torchaudio && python /opt/work/system1.py"


