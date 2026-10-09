#!/bin/bash

MOD=langchain

SIF=${MOD}.sif

SINGULARITY=apptainer

id=${1-"3"}
pyfile="system${id}.py"

MODEL_DIR=/home/kanazawa/models
if [ ! -d ${MODEL_DIR} ] ; then
    MODEL_DIR=/mnt/pool3/scratch/ishikawa/models
    if [ ! -d ${MODEL_DIR} ] ; then
	echo "Error : MODEL_DIR does not exist"
	exit -1
    fi
fi


${SINGULARITY} -q exec --no-home \
               --bind ${MODEL_DIR}:/opt/models \
               --bind $(pwd):/opt/work \
               --pwd /opt/work \
               --writable-tmpfs \
               ${SIF} bash -c \
               "export HF_HOME=/opt/work/hf_cache && pip uninstall -y torchaudio && python /opt/work/system${id}.py"


