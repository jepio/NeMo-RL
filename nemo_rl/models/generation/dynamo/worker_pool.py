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

"""Fixed Ray-managed pool of Dynamo vLLM subprocess owners."""

from collections import Counter
from collections.abc import Sequence
from typing import Any

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from nemo_rl.distributed.virtual_cluster import (
    DEFAULT_DYNAMO_KV_EVENT_PORT_RANGE_HIGH,
    DEFAULT_DYNAMO_KV_EVENT_PORT_RANGE_LOW,
    DEFAULT_DYNAMO_NIXL_PORT_RANGE_HIGH,
    DEFAULT_DYNAMO_NIXL_PORT_RANGE_LOW,
    DEFAULT_DYNAMO_SYSTEM_PORT_RANGE_HIGH,
    DEFAULT_DYNAMO_SYSTEM_PORT_RANGE_LOW,
    DEFAULT_VLLM_PORT_RANGE_LOW,
    DEFAULT_VLLM_PORTS_PER_ENGINE,
    RayVirtualCluster,
)
from nemo_rl.models.generation.dynamo.arguments import MultiNodeEngine
from nemo_rl.models.generation.dynamo.config import DynamoWorkerRole
from nemo_rl.models.generation.dynamo.dynamo_worker import (
    DynamoGpuReservation,
    DynamoVllmWorker,
)
from nemo_rl.models.generation.dynamo.venv import DYNAMO_WORKER_FQN
from nemo_rl.utils.venvs import make_actor_runtime_env


def _vllm_port_for_node_slot(node_slot: int) -> int:
    """Return vLLM's node-local scan base with 100 ports of headroom."""
    return DEFAULT_VLLM_PORT_RANGE_LOW + node_slot * DEFAULT_VLLM_PORTS_PER_ENGINE


def _nixl_port_for_node_slot(node_slot: int) -> int:
    """Return the managed node-local NIXL side-channel port.

    One port per data-parallel rank of an engine group is enough for vLLM 0.29:
    its NIXL scheduler serves every TP rank's handshake from one listener on
    ``VLLM_NIXL_SIDE_CHANNEL_PORT + data_parallel_index``. The pool advances the
    node slot by the data-parallel size, so engines on one node never share a port.
    """
    port = DEFAULT_DYNAMO_NIXL_PORT_RANGE_LOW + node_slot
    if port >= DEFAULT_DYNAMO_NIXL_PORT_RANGE_HIGH:
        raise ValueError(
            "Managed Dynamo has more node-local engine groups than its "
            f"NIXL-port band supports: slot={node_slot}"
        )
    return port


def _kv_event_port_for_node_slot(node_slot: int) -> int:
    """Return the managed node-local prefill KV-event port."""
    port = DEFAULT_DYNAMO_KV_EVENT_PORT_RANGE_LOW + node_slot
    if port >= DEFAULT_DYNAMO_KV_EVENT_PORT_RANGE_HIGH:
        raise ValueError(
            "Managed Dynamo has more node-local engine groups than its "
            f"KV-event-port band supports: slot={node_slot}"
        )
    return port


class FixedDynamoWorkerPool:
    """Reserve inference GPUs and launch one worker per model-parallel group.

    ``worker_roles`` is positional: element ``i`` is assigned to the ``i``-th
    engine group in placement-group iteration order. The same index sets that
    engine's NCCL refit rank offset (see ``DynamoRefitChannel``), so callers
    pass decode roles before prefill roles and keep the order stable. Port
    slots are node-local, while role indices are fleet-global.

    An engine whose TP x PP ranks do not fit on one node spans
    ``nodes_per_engine`` whole nodes. Its first node runs the leader, which
    serves the Dynamo endpoints and refit routes, and every other node runs a
    headless ``dynamo.vllm`` that only hosts vLLM workers.
    """

    def __init__(
        self,
        *,
        cluster: RayVirtualCluster,
        config: dict[str, Any],
        namespace: str,
        engine_world_size: int,
        worker_roles: Sequence[DynamoWorkerRole],
        manager_env: dict[str, str],
        startup_timeout_s: float,
        nodes_per_engine: int,
        data_parallel_size: int,
    ) -> None:
        self._cluster = cluster
        self._config = config
        self._namespace = namespace
        self._engine_world_size = engine_world_size
        self._nodes_per_engine = nodes_per_engine
        self._data_parallel_size = data_parallel_size
        self._worker_roles = tuple(worker_roles)
        if not self._worker_roles:
            raise ValueError("Managed Dynamo requires at least one worker role.")
        self._manager_env = manager_env
        self._startup_timeout_s = startup_timeout_s
        # One leader per engine. Refit and health checks address only leaders.
        self._workers: list[ray.actor.ActorHandle] = []
        # The headless workers on the other nodes of each multi-node engine.
        self._followers: list[ray.actor.ActorHandle] = []
        self._follower_cleanup_reservations: list[ray.actor.ActorHandle] = []
        self._reservations: list[ray.actor.ActorHandle] = []
        self._cleanup_reservations: list[ray.actor.ActorHandle] = []
        self._reservation_metadata: list[dict[str, Any]] = []
        self._metadata: list[dict[str, Any]] = []

    @property
    def size(self) -> int:
        return len(self._workers)

    def is_alive(self) -> bool:
        """Return whether every managed vLLM subprocess is still alive."""
        return bool(self._workers) and all(
            ray.get(
                [
                    worker.is_alive.remote()
                    for worker in (*self._workers, *self._followers)
                ]
            )
        )

    def _engine_node_groups(
        self, placement_groups: list[Any]
    ) -> list[list[tuple[Any, list[int]]]]:
        """Split the placement groups into engines.

        Returns:
            One list per engine. Each entry of that list is a placement group
            and the bundle indices of the engine on that node. Entry 0 is the
            leader node.
        """
        engines: list[list[tuple[Any, list[int]]]] = []
        if self._nodes_per_engine == 1:
            for pg_index, placement_group in enumerate(placement_groups):
                bundle_count = placement_group.bundle_count
                if bundle_count % self._engine_world_size != 0:
                    raise ValueError(
                        f"Inference placement group {pg_index} has {bundle_count} "
                        "GPU bundles, which is not divisible by engine_world_size="
                        f"{self._engine_world_size}."
                    )
                for start in range(0, bundle_count, self._engine_world_size):
                    engines.append(
                        [
                            (
                                placement_group,
                                list(range(start, start + self._engine_world_size)),
                            )
                        ]
                    )
            return engines
        gpus_per_node = self._engine_world_size // self._nodes_per_engine
        if len(placement_groups) % self._nodes_per_engine != 0:
            raise ValueError(
                f"{len(placement_groups)} inference nodes cannot be split into "
                f"engines of {self._nodes_per_engine} nodes."
            )
        for pg_index, placement_group in enumerate(placement_groups):
            if placement_group.bundle_count != gpus_per_node:
                raise ValueError(
                    f"Inference placement group {pg_index} has "
                    f"{placement_group.bundle_count} GPU bundles, but an engine "
                    f"of {self._nodes_per_engine} nodes uses all {gpus_per_node} "
                    "GPUs of each node."
                )
        for start in range(0, len(placement_groups), self._nodes_per_engine):
            engines.append(
                [
                    (placement_group, list(range(gpus_per_node)))
                    for placement_group in placement_groups[
                        start : start + self._nodes_per_engine
                    ]
                ]
            )
        return engines

    def start(self) -> None:
        if self._workers:
            raise RuntimeError("Managed Dynamo worker pool is already started.")
        placement_groups = self._cluster.get_placement_groups()
        # Builds the dynamo extra's venv on every node when uv-managed; the worker
        # actor launches dynamo.vllm with that venv's interpreter.
        runtime_env: dict[str, Any] = make_actor_runtime_env(DYNAMO_WORKER_FQN)

        group_index = 0
        role_indices: Counter[str] = Counter()
        engine_slots_by_node: dict[str, int] = {}
        system_ports_by_node: dict[str, set[int]] = {}
        metadata_refs: list[ray.ObjectRef] = []
        pg_indices = {
            id(placement_group): index
            for index, placement_group in enumerate(placement_groups)
        }
        for engine_nodes in self._engine_node_groups(placement_groups):
            if group_index >= len(self._worker_roles):
                raise ValueError(
                    "Inference placement groups contain more engine groups than "
                    f"the configured managed worker roles: {len(self._worker_roles)}"
                )
            worker_role = self._worker_roles[group_index]
            role_index = role_indices[worker_role]
            role_indices[worker_role] += 1
            group_name = f"{self._namespace}-dynamo-vllm-{worker_role}-{role_index}"
            # Every node of an engine uses the seed of its leader: the ranks of one
            # engine must sample alike.
            engine_seed = pg_indices[id(engine_nodes[0][0])] * 1024 + group_index

            # Reserve the GPUs of every node of this engine, then read where they are.
            node_plans: list[dict[str, Any]] = []
            for placement_group, bundle_indices in engine_nodes:
                reservation_handles = []
                for bundle_index in bundle_indices:
                    strategy = PlacementGroupSchedulingStrategy(
                        placement_group=placement_group,
                        placement_group_bundle_index=bundle_index,
                        placement_group_capture_child_tasks=True,
                    )
                    reservation_handles.append(
                        DynamoGpuReservation.options(
                            num_gpus=1,
                            runtime_env=runtime_env,
                            scheduling_strategy=strategy,
                        ).remote()
                    )
                self._reservations.extend(reservation_handles)
                reservation_metadata = ray.get(
                    [handle.metadata.remote() for handle in reservation_handles]
                )
                self._reservation_metadata.extend(reservation_metadata)
                node_ips = {item["node_ip"] for item in reservation_metadata}
                if len(node_ips) != 1:
                    raise RuntimeError(
                        "A placement group of a managed Dynamo engine spans multiple "
                        "nodes; a multi-node engine must use whole nodes."
                    )
                node_plans.append(
                    {
                        "placement_group": placement_group,
                        "bundle_indices": bundle_indices,
                        "reservations": reservation_handles,
                        "cuda_devices": [
                            item["gpu_id"] for item in reservation_metadata
                        ],
                        "node_ip": next(iter(node_ips)),
                    }
                )
            if len({plan["node_ip"] for plan in node_plans}) != len(node_plans):
                raise RuntimeError(
                    "Two nodes of a multi-node Dynamo engine share a host: "
                    f"{[plan['node_ip'] for plan in node_plans]}."
                )

            multi_node_master: tuple[str, int] | None = None
            if len(node_plans) > 1:
                leader = node_plans[0]
                leader_ports = system_ports_by_node.setdefault(leader["node_ip"], set())
                master_port = ray.get(
                    leader["reservations"][0].select_free_port.remote(
                        port_range_low=DEFAULT_DYNAMO_SYSTEM_PORT_RANGE_LOW,
                        port_range_high=DEFAULT_DYNAMO_SYSTEM_PORT_RANGE_HIGH,
                        excluded_ports=sorted(leader_ports),
                    )
                )
                leader_ports.add(master_port)
                multi_node_master = (leader["node_ip"], master_port)

            for node_rank, plan in enumerate(node_plans):
                node_ip = plan["node_ip"]
                node_slot = engine_slots_by_node.get(node_ip, 0)
                # A data-parallel engine uses one NIXL and KV-event port per rank,
                # counted up from the port of its slot.
                engine_slots_by_node[node_ip] = node_slot + self._data_parallel_size
                system_ports = system_ports_by_node.setdefault(node_ip, set())
                system_port = ray.get(
                    plan["reservations"][0].select_free_port.remote(
                        port_range_low=DEFAULT_DYNAMO_SYSTEM_PORT_RANGE_LOW,
                        port_range_high=DEFAULT_DYNAMO_SYSTEM_PORT_RANGE_HIGH,
                        excluded_ports=sorted(system_ports),
                    )
                )
                system_ports.add(system_port)
                nixl_port = (
                    _nixl_port_for_node_slot(node_slot)
                    if worker_role != "aggregated"
                    else None
                )
                kv_event_port = (
                    _kv_event_port_for_node_slot(node_slot)
                    if worker_role == "prefill"
                    else None
                )
                multi_node = (
                    MultiNodeEngine(
                        nnodes=len(node_plans),
                        node_rank=node_rank,
                        master_addr=multi_node_master[0],
                        master_port=multi_node_master[1],
                    )
                    if multi_node_master is not None
                    else None
                )
                node_group_name = (
                    group_name if node_rank == 0 else f"{group_name}-node{node_rank}"
                )
                worker = DynamoVllmWorker.options(
                    num_gpus=0,
                    runtime_env={
                        **runtime_env,
                        "env_vars": {
                            **runtime_env["env_vars"],
                            "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                        },
                    },
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=plan["placement_group"],
                        placement_group_bundle_index=plan["bundle_indices"][0],
                        placement_group_capture_child_tasks=True,
                    ),
                    name=node_group_name,
                ).remote(
                    self._config,
                    namespace=self._namespace,
                    group_name=node_group_name,
                    cuda_devices=plan["cuda_devices"],
                    system_port=system_port,
                    vllm_port=_vllm_port_for_node_slot(node_slot),
                    nixl_port=nixl_port,
                    kv_event_port=kv_event_port,
                    worker_role=worker_role,
                    manager_env=self._manager_env,
                    startup_timeout_s=self._startup_timeout_s,
                    seed=engine_seed,
                    cleanup_reservation=plan["reservations"][0],
                    multi_node=multi_node,
                )
                if node_rank == 0:
                    self._workers.append(worker)
                    self._cleanup_reservations.append(plan["reservations"][0])
                    self._metadata.append({})
                    metadata_refs.append(worker.metadata.remote())
                else:
                    self._followers.append(worker)
                    self._follower_cleanup_reservations.append(plan["reservations"][0])
            group_index += 1

        if group_index != len(self._worker_roles):
            raise ValueError(
                "Inference placement groups contain fewer engine groups than the "
                f"configured managed worker roles: placed={group_index}, "
                f"configured={len(self._worker_roles)}"
            )

        metadata_error: Exception | None = None
        for index, metadata_ref in enumerate(metadata_refs):
            try:
                self._metadata[index] = dict(ray.get(metadata_ref))
            except Exception as error:
                if metadata_error is None:
                    metadata_error = error
        if metadata_error is not None:
            raise metadata_error
        # A headless worker has no endpoint to wait for. The leader only starts once
        # every node has joined, so a follower that died would already have failed it.
        if self._followers and not all(
            ray.get([worker.is_alive.remote() for worker in self._followers])
        ):
            raise RuntimeError("A headless Dynamo vLLM worker exited during startup.")

    def refit_workers(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self._metadata]

    def validate(self, expected: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self._workers or not all(
            ray.get([w.is_alive.remote() for w in (*self._workers, *self._followers)])
        ):
            raise RuntimeError("A Ray-managed Dynamo vLLM worker exited.")
        try:
            current_reservations = ray.get(
                [reservation.metadata.remote() for reservation in self._reservations]
            )
        except Exception as exc:
            raise RuntimeError(
                "A Ray-managed Dynamo GPU reservation actor exited."
            ) from exc
        if current_reservations != self._reservation_metadata:
            raise RuntimeError(
                "Ray-managed Dynamo GPU reservation membership changed: "
                f"expected={self._reservation_metadata}, current={current_reservations}."
            )
        current = ray.get([worker.metadata.remote() for worker in self._workers])
        if current != expected:
            raise RuntimeError(
                "Ray-managed Dynamo worker membership changed after NCCL collective "
                f"initialization: expected={expected}, current={current}."
            )
        return [dict(item) for item in current]

    def shutdown(self) -> None:
        for worker, reservation in zip(
            (*self._workers, *self._followers),
            (*self._cleanup_reservations, *self._follower_cleanup_reservations),
            strict=True,
        ):
            try:
                ray.get(worker.shutdown.remote(), timeout=30)
            except Exception:
                try:
                    ray.get(
                        reservation.cleanup_process_group.remote(),
                        timeout=15,
                    )
                except Exception:
                    pass
        for worker in (*self._workers, *self._followers):
            try:
                ray.kill(worker, no_restart=True)
            except Exception:
                pass
        for reservation in self._reservations:
            try:
                ray.kill(reservation, no_restart=True)
            except Exception:
                pass
        self._workers.clear()
        self._followers.clear()
        self._follower_cleanup_reservations.clear()
        self._reservations.clear()
        self._cleanup_reservations.clear()
        self._reservation_metadata.clear()
        self._metadata.clear()
