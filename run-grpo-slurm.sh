#!/bin/bash
# Run from the root of NeMo RL repo
NUM_ACTOR_NODES=1

# grpo_math_8b uses Llama-3.1-8B-Instruct model
COMMAND="./run-grpo.sh" \
CONTAINER=./nvcr.io/nvidia/nemo-rl:v0.6.0 \
MOUNTS="$PWD:$PWD" \
sbatch \
    --nodes=${NUM_ACTOR_NODES} \
    --account=coreai_mlperf_training \
    --job-name=coreai_mlperf_training-grpo.nemotron9b-workplace-assistant \
    --partition=batch_short \
    --time=2:0:0 \
    --gres=gpu:8 \
    ray.sub
