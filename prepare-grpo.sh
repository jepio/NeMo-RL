#!/bin/bash

export HF_HOME=$PWD/.cache

if [ -z "${HF_TOKEN:-}" ]; then
  echo >&2 "set HF_TOKEN"
  exit 1
fi

git submodule update --init --recursive

## Prepare container image
CONTAINER_IMAGE_PATH=nvcr.io/nvidia/nemo-rl:v0.6.0
if [ ! -f ./${CONTAINER_IMAGE_PATH} ]; then
  mkdir -p "$(dirname "$CONTAINER_IMAGE_PATH")"
  enroot import -o "$CONTAINER_IMAGE_PATH" "docker://${CONTAINER_IMAGE_PATH}"
fi
# Swap to local container path
CONTAINER_IMAGE_PATH=./$CONTAINER_IMAGE_PATH

## Fetch Gym dataset jsonl
cd 3rdparty/Gym-workspace/Gym
uv venv --python 3.12 --allow-existing .venv
source .venv/bin/activate
uv sync --active --extra dev

if [ ! -f env.yaml ]; then
  echo "hf_token: ${HF_TOKEN}" >env.yaml
fi
# aligned with the training we are going to be doing
config_paths="responses_api_models/vllm_model/configs/vllm_model_for_training.yaml,\
resources_servers/workplace_assistant/configs/workplace_assistant.yaml"
uv run ng_prepare_data "+config_paths=[${config_paths}]" \
    +output_dirpath=data/workplace_assistant \
    +mode=train_preparation \
    +should_download=true \
    +data_source=huggingface

deactivate
cd ../../..

## Fetch Model
hf download nvidia/NVIDIA-Nemotron-Nano-9B-v2

# this is evil but necessary
tokenizer_config_path=$(find $PWD/.cache/hub/models--nvidia--NVIDIA-Nemotron-Nano-9B-v2 -name tokenizer_config.json)
sed -i 's/enable_thinking=true/enable_thinking=false/g' $tokenizer_config_path
sed -i 's/{%- if messages\[-1\]\['\''role'\''\] == '\''assistant'\'' -%}{%- set ns.last_turn_assistant_content = messages\[-1\]\['\''content'\''\].strip() -%}{%- set messages = messages\[:-1\] -%}{%- endif -%}//g' $tokenizer_config_path

