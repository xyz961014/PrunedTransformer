#!/bin/bash

CMD=${1:-/bin/bash}
NV_VISIBLE_DEVICES=${2:-"all"}
DOCKER_BRIDGE=${3:-"host"}

docker run -it --rm \
  --gpus device=$NV_VISIBLE_DEVICES \
  --net=$DOCKER_BRIDGE \
  --shm-size=1g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e LD_LIBRARY_PATH='/workspace/install/lib/:/usr/local/python3/lib' \
  -e NVIDIA_BUILD_ID='12345' \
  -e BERT_PREP_WORKING_DIR='/workspace/bert/data' \
  -v $PWD:/workspace/bert \
  -v /home/xyz/Documents/experiment/bert:/results \
  -v /home/xyz/Documents/Dataset:/data \
  bert $CMD
