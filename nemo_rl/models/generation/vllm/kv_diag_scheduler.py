# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""A vLLM scheduler that logs where the KV cache blocks of an engine sit.

Use it to study the KV memory of long multi-turn rollouts. Set
``vllm_kwargs.scheduler_cls`` to
``nemo_rl.models.generation.vllm.kv_diag_scheduler.KvDiagScheduler``. Every
``NRL_KV_DIAG_INTERVAL_S`` seconds (default 15) the engine core logs one
``[KV-DIAG]`` line with the blocks by state:

* ``free``: blocks that are free or can be evicted, ``cached_free`` of them hold a
  prefix that a later request can still hit;
* ``in_use``: blocks with a reference, split by owner into running requests,
  waiting requests, and finished requests whose blocks are held (for example
  until a decode engine pulls them);
* the size of every connector container that looks like a transfer queue.

The scheduler keeps the behavior of the cadenced prefill scheduler when the
``vllm_extensions`` package is importable and of vLLM's async scheduler
otherwise.
"""

import os
import time
from typing import Any

from vllm.logger import init_logger

try:
    from vllm_extensions.prefill_scheduler import (
        CadencedPrefillAsyncScheduler as _BaseScheduler,
    )
except ImportError:
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler as _BaseScheduler

logger = init_logger(__name__)

_TRANSFER_NAME_HINTS = (
    "need",
    "send",
    "lease",
    "expire",
    "transfer",
    "pending",
    "hold",
    "delay",
    "finished",
    "recv",
)


def _block_ids(blocks: Any) -> set[int]:
    return {block.block_id for block in blocks if not getattr(block, "is_null", False)}


def summarize_scheduler(scheduler: Any) -> dict[str, Any]:
    """Count the KV blocks of ``scheduler`` by state and owner.

    Reads only attributes that vLLM keeps for its own bookkeeping, so it never
    changes the scheduler.
    """
    manager = scheduler.kv_cache_manager
    pool = manager.block_pool
    blocks = pool.blocks
    live = [block for block in blocks if not getattr(block, "is_null", False)]
    cached = [block for block in live if block.block_hash is not None]
    summary: dict[str, Any] = {
        "total": pool.num_gpu_blocks,
        "free": pool.get_num_free_blocks(),
        "in_use": sum(1 for block in live if block.ref_cnt > 0),
        "cached": len(cached),
        "cached_free": sum(1 for block in cached if block.ref_cnt == 0),
        "cached_in_use": sum(1 for block in cached if block.ref_cnt > 0),
        "running_reqs": len(scheduler.running),
        "waiting_reqs": len(scheduler.waiting) + len(scheduler.skipped_waiting),
        "requests": len(scheduler.requests),
    }

    running_ids = {request.request_id for request in scheduler.running}
    owners: dict[str, set[int]] = {"running": set(), "other": set()}
    finished_ids: set[str] = set()
    per_group: dict[str, set[int]] = {}
    groups = scheduler.kv_cache_config.kv_cache_groups
    for index, group_manager in enumerate(manager.coordinator.single_type_managers):
        spec_name = type(groups[index].kv_cache_spec).__name__
        for request_id, request_blocks in group_manager.req_to_blocks.items():
            ids = _block_ids(request_blocks)
            per_group.setdefault(f"{index}:{spec_name}", set()).update(ids)
            if request_id in running_ids:
                owners["running"].update(ids)
                continue
            request = scheduler.requests.get(request_id)
            if request is not None and request.is_finished():
                finished_ids.add(request_id)
            owners["other"].update(ids)
    summary["running_blocks"] = len(owners["running"])
    summary["held_other_blocks"] = len(owners["other"])
    summary["held_finished_reqs"] = len(finished_ids)
    summary["groups"] = {name: len(ids) for name, ids in per_group.items()}

    connector = getattr(scheduler, "connector", None)
    queues: dict[str, int] = {}
    for holder in (connector, getattr(connector, "connector_scheduler", None)):
        if holder is None:
            continue
        for name, value in vars(holder).items():
            if isinstance(value, (dict, set, list)) and any(
                hint in name.lower() for hint in _TRANSFER_NAME_HINTS
            ):
                queues[name] = len(value)
    summary["connector"] = queues
    return summary


class KvDiagScheduler(_BaseScheduler):
    """Log ``summarize_scheduler`` every ``NRL_KV_DIAG_INTERVAL_S`` seconds."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._kv_diag_interval_s = float(os.environ.get("NRL_KV_DIAG_INTERVAL_S", "15"))
        self._kv_diag_last = 0.0
        self._kv_diag_failed = False

    def schedule(self, *args: Any, **kwargs: Any):
        output = super().schedule(*args, **kwargs)
        now = time.monotonic()
        if (
            not self._kv_diag_failed
            and now - self._kv_diag_last >= self._kv_diag_interval_s
        ):
            self._kv_diag_last = now
            try:
                summary = summarize_scheduler(self)
            except Exception:
                # Diagnostics must never break an engine: report once and stop.
                self._kv_diag_failed = True
                logger.exception("[KV-DIAG] disabled: could not read the KV state")
            else:
                transfer = self.vllm_config.kv_transfer_config
                logger.info(
                    "[KV-DIAG] role=%s dp_rank=%s %s",
                    getattr(transfer, "kv_role", None) if transfer else None,
                    self.vllm_config.parallel_config.data_parallel_rank,
                    " ".join(f"{key}={value}" for key, value in summary.items()),
                )
        return output
