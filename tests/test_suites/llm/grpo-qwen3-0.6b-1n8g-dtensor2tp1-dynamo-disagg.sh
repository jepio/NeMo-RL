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

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
source "${SCRIPT_DIR}/common.env"

# ===== BEGIN CONFIG =====
NUM_NODES=1
GPUS_PER_NODE=8
STEPS_PER_RUN=2
MAX_STEPS=2
NUM_RUNS=1
NUM_MINUTES=30
# ===== END CONFIG =====

# The H100 suite allocates eight GPUs; the recipe uses only three.
exit_if_max_steps_reached

# Managed Dynamo runs from the dynamo extra's actor venv (vLLM + ai-dynamo + nemo_rl).
dynamo_python=$(uv run --no-sync python -c \
  'from nemo_rl.models.generation.dynamo.venv import get_dynamo_python; print(get_dynamo_python())' | tail -n 1)
"${dynamo_python}" -c \
  'import importlib.metadata as m; assert m.version("ai-dynamo") == "1.5.1"; assert m.version("vllm") == "0.29.0"; assert m.version("nixl") == "1.3.0"; assert m.version("nixl-cu13") == "1.3.0"'
uv run --no-sync python -c \
  'from nemo_rl.models.generation.dynamo.venv import get_dynamo_service_executable as e; e("etcd"); e("nats-server")'

cd "${PROJECT_ROOT}"
uv run --no-sync \
  "${PROJECT_ROOT}/examples/run_grpo.py" \
  --config "${CONFIG_PATH}" \
  grpo.max_num_steps="${MAX_STEPS}" \
  logger.log_dir="${LOG_DIR}" \
  2>&1 | tee "${RUN_LOG}"

grep -F "'--disaggregation-mode', 'decode'" "${RUN_LOG}"
grep -F "'--disaggregation-mode', 'prefill'" "${RUN_LOG}"
grep -F "'VLLM_NIXL_SIDE_CHANNEL_PORT':" "${RUN_LOG}"
component_ready_line=$(grep -F \
    "frontend ready with managed components" "${RUN_LOG}")
grep -Fq "'backend': 1" <<< "${component_ready_line}"
grep -Fq "'prefill': 1" <<< "${component_ready_line}"
# Prove decode pulled KV from prefill rather than prefilling locally. vLLM logs this
# only in the decode-side NIXL handshake, which runs only for requests with
# remote-prefill kv_transfer_params (vllm/distributed/kv_transfer/kv_connector/v1/nixl).
grep -F "NIXL compatibility check passed" "${RUN_LOG}"

# The recipe refits in flight without clearing caches: vLLM cannot clear them while
# decode requests await remote KV. Every refit must be bracketed by a keep-mode
# pause and resume on both prefill and decode engines, with no drain or flush.
grep -F "Performing policy generation refit" "${RUN_LOG}"
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

uv run --no-sync tests/json_dump_tb_logs.py \
  "${LOG_DIR}" \
  --output_path "${JSON_METRICS}" \
  --require-tag-prefix "generation_metrics/"
last_step=$(jq '[.["train/loss"] | keys[] | tonumber] | max' "${JSON_METRICS}")
if [[ "${last_step}" != "${MAX_STEPS}" ]]; then
  echo "Expected ${MAX_STEPS} training steps, found ${last_step}" >&2
  exit 1
fi
uv run --no-sync tests/check_metrics.py \
  "${JSON_METRICS}" \
  'max(data["train/token_mult_prob_error"]) < 1.05'

if pgrep -f "[d]ynamo.frontend|[d]ynamo.vllm|[V]LLM::EngineCore|[n]emorl_dynamo_etcd_|[n]emorl_dynamo_nats_"; then
  echo "Managed Dynamo processes remain after GRPO shutdown" >&2
  exit 1
fi
