#!/bin/bash

# reinstall ray so that dashboard works:
uv pip install --link-mode=copy --force-reinstall "ray[default]==2.54.0"

# noisy error in workplace_assistant
export PYTHONWARNINGS="ignore:A value is trying to be set on a copy of a slice from a DataFrame:Warning"

# Set experiment name with timestamp
EXP_NAME="$(date +%Y%m%d)/nemo_gym_grpo/nemotron_nano_v2_9b/workplace_assistant_001"
# Resume from checkpoint
#EXP_NAME=20260611/nemo_gym_grpo/nemotron_nano_v2_9b/workplace_assistant_001

mkdir -p results/$EXP_NAME

# Configuration file path
CONFIG_PATH=examples/nemo_gym/grpo_workplace_assistant_nemotron_nano_v2_9b.yaml

# Launch training
# Set these environment variables before running:
#   WANDB_API_KEY: Your Weights & Biases API key for logging
#   logger.wandb.project: Fill in your username
#
# This is the original, needed to be patched for vllm 0.17.1 compat
# ++policy.generation.vllm_cfg.tool_parser_plugin=$(find $PWD/.cache -name nemotron_toolcall_parser_no_streaming.py)
# ++grpo.max_num_steps=3

TORCH_CUDA_ARCH_LIST="9.0 10.0" \
HF_HOME=$PWD/.cache/ \
uv run python examples/nemo_gym/run_grpo_nemo_gym.py \
    --config=$CONFIG_PATH \
    ++logger.log_dir=results/$EXP_NAME \
    ++logger.wandb_enabled=False \
    ++policy.generation.vllm_cfg.tool_parser_plugin=$PWD/examples/nemo_gym/nemotron_toolcall_parser_vllm0171.py \
    ++grpo.max_num_steps=20 \
    ++grpo.max_num_epochs=1 \
    ++grpo.val_period=5 \
    ++grpo.val_at_end=true \
    ++checkpointing.save_period=19 \
    ++checkpointing.save_optimizer=False \
    ++policy.sequence_packing.enabled=True \
    ++policy.dynamic_batching.enabled=False \
    ++policy.megatron_cfg.defer_fp32_logits=True \
    ++policy.logprob_batch_size=2 \
    ++loss_fn.force_on_policy_ratio=True \
    ++checkpointing.checkpoint_dir=results/$EXP_NAME 2>&1 | tee results/$EXP_NAME/output.log
