#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath "${SCRIPT_DIR}/../..")
EXP_NAME=$(basename "$0" .sh)
EXP_DIR=${SCRIPT_DIR}/${EXP_NAME}
LOG_DIR=${EXP_DIR}/logs
RUN_LOG=${EXP_DIR}/run.log
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-}

rm -rf "${EXP_DIR}"
mkdir -p "${LOG_DIR}"
git config --global --add safe.directory "${PROJECT_ROOT}"

# Two-GPU smoke for in-flight weight updates on managed Dynamo: every refit freezes
# the engine's requests with vLLM's keep-mode pause instead of draining them.
#
# Managed Dynamo runs from the dynamo extra's actor venv (vLLM + ai-dynamo + nemo_rl),
# built here if needed; etcd and nats-server come from NEMO_RL_DYNAMO_BIN_DIR or PATH.
dynamo_python=$(uv run --no-sync python -c \
  'from nemo_rl.models.generation.dynamo.venv import get_dynamo_python; print(get_dynamo_python())' | tail -n 1)
"${dynamo_python}" -c \
  'import importlib.metadata as m; assert m.version("ai-dynamo") == "1.5.1"; assert m.version("vllm") == "0.29.0"; assert m.version("nixl") == "1.3.0"; assert m.version("nixl-cu13") == "1.3.0"'
uv run --no-sync python -c \
  'from nemo_rl.models.generation.dynamo.venv import get_dynamo_service_executable as e; e("etcd"); e("nats-server")'

cd "${PROJECT_ROOT}"
uv run --no-sync coverage run -a \
  --data-file="${PROJECT_ROOT}/tests/.coverage" \
  --source="${PROJECT_ROOT}/nemo_rl" \
  "${PROJECT_ROOT}/examples/run_grpo.py" \
  --config "${PROJECT_ROOT}/examples/configs/grpo_math_1B_dynamo.yaml" \
  policy.model_name=Qwen/Qwen3-0.6B \
  policy.tokenizer.name=Qwen/Qwen3-0.6B \
  logger.log_dir="${LOG_DIR}" \
  grpo.async_grpo.in_flight_weight_updates=true \
  grpo.async_grpo.recompute_kv_cache_after_weight_updates=true \
  2>&1 | tee "${RUN_LOG}"

grep -F "Performing policy generation refit" "${RUN_LOG}"

# Each refit must be bracketed by a keep-mode pause and resume. Clearing the cache
# happens at the pause, so no separate post-refit invalidation may run.
refit_count=$(grep -Fc "Performing policy generation refit" "${RUN_LOG}" || true)
pause_count=$(grep -Fc "Requesting dynamo generation pause before refit" "${RUN_LOG}" || true)
resume_count=$(grep -Fc "Requesting dynamo generation resume after refit" "${RUN_LOG}" || true)
if [[ "${refit_count}" -eq 0 || "${pause_count}" -ne "${refit_count}" || "${resume_count}" -ne "${refit_count}" ]]; then
  echo "Expected one pause and one resume per refit; refits=${refit_count}, pauses=${pause_count}, resumes=${resume_count}" >&2
  exit 1
fi
if grep -Fq \
  -e "Invalidated generation backend KV caches after weight update" \
  -e "Failed to invalidate generation backend KV caches" \
  -e "Non-async engine: waiting for all pending generations" \
  "${RUN_LOG}"; then
  echo "An in-flight refit drained or flushed instead of pausing" >&2
  exit 1
fi
if grep -Fq "Failed to resume dynamo generation" "${RUN_LOG}"; then
  echo "Dynamo failed to resume generation after a refit" >&2
  exit 1
fi

metrics_json=${EXP_DIR}/metrics.json
uv run --no-sync tests/json_dump_tb_logs.py \
  "${LOG_DIR}" \
  --output_path "${metrics_json}" \
  --require-tag-prefix "generation_metrics/"
uv run --no-sync tests/check_metrics.py \
  "${metrics_json}" \
  'max(data["train/token_mult_prob_error"]) < 1.05'

if pgrep -f "[d]ynamo.frontend|[d]ynamo.vllm|[V]LLM::EngineCore|[n]emorl_dynamo_etcd_|[n]emorl_dynamo_nats_"; then
  echo "Managed Dynamo processes remain after GRPO shutdown" >&2
  exit 1
fi
