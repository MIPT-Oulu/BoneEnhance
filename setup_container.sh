#!/bin/bash

MODEL=$1  # Name of the image
NAME=$2  # Name for experiment - used to store preprocessed images
LOCAL_PATH_TO_IMAGES=$3  # Attach as volume - where the images are stored
LOCAL_PATH_TO_PREPROCESSED_IMAGES=$4 # Attach as volume - where the preprocessed images will be stored. They will be stored under the NAME_EXPERIMENT pattern
LOCAL_PICKLE_FILE=$5  # Copy into container before generating predictions - where the pickle file is located
LOCAL_PREDICTION_PATH=$6  # Copy container prediction file to local prediction file
DEVICE=$7  # gpu or cpu
GPU=$8  # GPU to execute on
BOOTSTRAPPING=$9  # whether to use bootstrapping

echo "Checking if users.txt exists"
# Check if users.txt exists. It should be created from template_users.txt.
if [[ ! -f users.txt ]]; then
    echo "users.txt does not exist, please check README.md on how to create it. Exiting."
    exit
fi

get_groupname () {
    ENTRIES=()

    while read line; do
        if [[ ! -z "$line" ]]
        then
            ENTRIES+=("$line")
        fi
    done < users.txt

    GROUP_INFO=($(echo ${ENTRIES[0]} | tr "," " "))
    GROUPNAME=${GROUP_INFO[0]}
    GROUPID=${GROUP_INFO[1]}

    echo ${GROUPNAME}
}

build_docker_image () {
    GROUPNAME=$(get_groupname)
    echo "${GROUPNAME}"
    echo "Building image"
    docker build --debug -t "${MODEL}" -f Dockerfile --build-arg GROUPNAME="${GROUPNAME}" .
}

run_model () {
  echo "Running model ${MODEL}"
}

if docker images | grep -q -w "${MODEL}"; then
    echo "Model already built, running."
    run_model
else
    echo "Model not built, building."
    build_docker_image
    run_model
fi