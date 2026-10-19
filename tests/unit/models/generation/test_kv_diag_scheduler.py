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

import importlib
import sys
import types
from types import SimpleNamespace

import pytest


@pytest.fixture
def kv_diag(monkeypatch):
    """Import the module against a minimal stand-in for vLLM."""

    class FakeBase:
        def __init__(self, *args, **kwargs):
            self.scheduled = 0

        def schedule(self, *args, **kwargs):
            self.scheduled += 1
            return "output"

    logger = SimpleNamespace(
        info=lambda *args, **kwargs: None,
        exception=lambda *args, **kwargs: None,
    )
    vllm = types.ModuleType("vllm")
    vllm_logger = types.ModuleType("vllm.logger")
    vllm_logger.init_logger = lambda name: logger
    extensions = types.ModuleType("vllm_extensions")
    prefill = types.ModuleType("vllm_extensions.prefill_scheduler")
    prefill.CadencedPrefillAsyncScheduler = FakeBase
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.logger", vllm_logger)
    monkeypatch.setitem(sys.modules, "vllm_extensions", extensions)
    monkeypatch.setitem(sys.modules, "vllm_extensions.prefill_scheduler", prefill)
    name = "nemo_rl.models.generation.vllm.kv_diag_scheduler"
    monkeypatch.delitem(sys.modules, name, raising=False)
    return importlib.import_module(name)


def _block(block_id, ref_cnt=0, block_hash=None, is_null=False):
    return SimpleNamespace(
        block_id=block_id, ref_cnt=ref_cnt, block_hash=block_hash, is_null=is_null
    )


def _scheduler():
    blocks = [
        _block(0, is_null=True),
        _block(1, ref_cnt=1, block_hash="a"),  # running, cached
        _block(2, ref_cnt=1),  # running, not cached
        _block(3, ref_cnt=1, block_hash="b"),  # held by a finished request
        _block(4, block_hash="c"),  # cached and evictable
        _block(5),  # free
    ]
    pool = SimpleNamespace(
        num_gpu_blocks=len(blocks),
        blocks=blocks,
        get_num_free_blocks=lambda: 2,
    )
    attention = SimpleNamespace(
        req_to_blocks={"run": [blocks[1], blocks[2]], "done": [blocks[3]]}
    )
    mamba = SimpleNamespace(req_to_blocks={"run": [blocks[2]]})
    manager = SimpleNamespace(
        block_pool=pool,
        coordinator=SimpleNamespace(single_type_managers=[attention, mamba]),
    )
    finished = SimpleNamespace(is_finished=lambda: True)
    running = SimpleNamespace(request_id="run", is_finished=lambda: False)
    connector = SimpleNamespace(
        _reqs_need_send={"done": 1.0}, _reqs_need_recv={}, side_channel_port=5600
    )
    return SimpleNamespace(
        kv_cache_manager=manager,
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(kv_cache_spec=SimpleNamespace()),
                SimpleNamespace(kv_cache_spec=SimpleNamespace()),
            ]
        ),
        running=[running],
        waiting=[object()],
        skipped_waiting=[],
        requests={"run": running, "done": finished},
        connector=connector,
    )


def test_summarize_scheduler_counts_blocks_by_state_and_owner(kv_diag) -> None:
    summary = kv_diag.summarize_scheduler(_scheduler())

    assert summary["total"] == 6
    assert summary["free"] == 2
    assert summary["in_use"] == 3
    assert summary["cached"] == 3
    assert summary["cached_free"] == 1
    assert summary["cached_in_use"] == 2
    assert summary["running_reqs"] == 1
    assert summary["waiting_reqs"] == 1
    assert summary["running_blocks"] == 2
    assert summary["held_other_blocks"] == 1
    assert summary["held_finished_reqs"] == 1
    assert summary["groups"] == {"0:SimpleNamespace": 3, "1:SimpleNamespace": 1}
    # Only containers that look like a transfer queue are reported.
    assert summary["connector"] == {"_reqs_need_send": 1, "_reqs_need_recv": 0}


def test_kv_diag_scheduler_reports_on_an_interval_and_never_breaks(
    kv_diag, monkeypatch
) -> None:
    monkeypatch.setenv("NRL_KV_DIAG_INTERVAL_S", "0")
    scheduler = kv_diag.KvDiagScheduler()
    scheduler.vllm_config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_role="kv_both"),
        parallel_config=SimpleNamespace(data_parallel_rank=0),
    )
    assert scheduler.schedule() == "output"
    # The fake scheduler has no KV state: the report fails once and then stops.
    assert scheduler._kv_diag_failed is True
    assert scheduler.schedule() == "output"
    assert scheduler.scheduled == 2
