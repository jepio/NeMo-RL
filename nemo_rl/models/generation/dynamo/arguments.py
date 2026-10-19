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

"""Deterministic argument and environment construction for managed Dynamo."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from nemo_rl.models.generation.dynamo.config import (
    DYNAMO_VLLM_FLAGS,
    DynamoCfg,
    DynamoWorkerRole,
)


@dataclass(frozen=True)
class MultiNodeEngine:
    """The place of one node in a vLLM engine that spans several nodes.

    Node rank 0 is the leader: it serves Dynamo endpoints and starts the engine
    core. Every other node runs ``dynamo.vllm --headless``, which starts vLLM
    workers only and joins the leader at ``master_addr:master_port``.
    """

    nnodes: int
    node_rank: int
    master_addr: str
    master_port: int
    # None: the ranks are TP x PP ranks that span the nodes. A number: the engine
    # has data-parallel ranks, this many per node, and the nodes join through
    # vLLM's data-parallel launch (address, RPC port and start rank) instead.
    data_parallel_size_local: int | None = None


# vllm_cfg settings that a prefill or decode role may replace through
# disaggregation.{prefill,decode}_vllm_kwargs. Parallelism and dtype stay shared.
_ROLE_OVERRIDABLE_CFG_KEYS = ("gpu_memory_utilization",)

_MANAGED_FLAGS = {
    "--data-parallel-size-local",
    "--data-parallel-address",
    "--data-parallel-rpc-port",
    "--data-parallel-start-rank",
    "--nnodes",
    "--node-rank",
    "--master-addr",
    "--master-port",
    "--headless",
    "--component",
    "--disaggregation-mode",
    "--model-name",
    "--model-path",
    "--endpoint",
    # Refit loads NeMo RL's vLLM worker extension and addresses each engine's
    # ranks as TP x PP; extra_cli_args must not replace or reshape either.
    "--worker-extension-cls",
    "--weight-transfer-config",
    "--data-parallel-size",
    "--data-parallel-size-local",
    "--distributed-executor-backend",
    "--kv-events-config",
    "--kv-transfer-config",
}

# NeMo RL's vLLM worker extension, which every managed engine loads so the driver
# can drive refit through Dynamo's /engine/* RL routes.
DYNAMO_WORKER_EXTENSION_CLS = (
    "nemo_rl.models.generation.vllm.dynamo_worker_extension.DynamoWorkerExtension"
)

_CREDENTIAL_FLAGS = {
    "--access-token",
    "--api-key",
    "--apikey",
    "--auth-token",
    "--hf-token",
    "--password",
    "--secret",
    "--token",
}


def _normalise_flag(flag: str) -> str:
    flag = flag.split("=", 1)[0].replace("_", "-")
    if flag.startswith("--no-"):
        return "--" + flag[5:]
    return flag


def _flag_for_key(key: str) -> str:
    return "--" + key.replace("_", "-")


def _serialise_value(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, separators=(",", ":"), sort_keys=True)
    return str(value)


class _ArgvBuilder:
    def __init__(self) -> None:
        self.argv: list[str] = []
        self.sources: dict[str, str] = {}

    def add(self, flag: str, value: Any = None, *, source: str) -> None:
        normalised = _normalise_flag(flag)
        prior = self.sources.get(normalised)
        if prior is not None:
            raise ValueError(
                f"Dynamo worker option {normalised} is set by both {prior} and {source}."
            )
        self.sources[normalised] = source
        if isinstance(value, bool):
            self.argv.append(flag if value else f"--no-{flag.removeprefix('--')}")
        else:
            self.argv.append(flag)
            if value is not None:
                self.argv.append(_serialise_value(value))

    def add_raw(self, args: Sequence[str], *, source: str) -> None:
        if not args:
            return
        current_flag: str | None = None
        current_has_value = False
        for token in args:
            if token.startswith("--"):
                current_flag = token
                current_has_value = "=" in token
                normalised = _normalise_flag(token)
                if normalised in _MANAGED_FLAGS:
                    raise ValueError(
                        f"{source} may not override managed option {normalised}."
                    )
                if normalised in self.sources:
                    raise ValueError(
                        f"Dynamo worker option {normalised} is set by both "
                        f"{self.sources[normalised]} and {source}."
                    )
                self.sources[normalised] = source
            elif current_flag is None or current_has_value:
                raise ValueError(
                    f"{source} contains invalid positional argument {token!r}; "
                    "values must immediately follow one --option."
                )
            else:
                current_has_value = True
        self.argv.extend(args)


def build_dynamo_vllm_argv(
    *,
    model_name: str,
    namespace: str,
    seed: int,
    vllm_cfg: Mapping[str, Any],
    vllm_kwargs: Mapping[str, Any],
    dynamo_cfg: DynamoCfg,
    worker_role: DynamoWorkerRole,
    kv_event_port: int | None,
    multi_node: MultiNodeEngine | None = None,
) -> list[str]:
    """Build a conflict-free ``python -m dynamo.vllm`` argument list.

    ``kv_event_port`` must be set for ``prefill`` workers and must be ``None``
    for ``decode`` and ``aggregated`` workers. ``multi_node`` is set for each
    node of an engine whose TP x PP ranks span several nodes.
    """
    builder = _ArgvBuilder()
    if multi_node is not None and multi_node.data_parallel_size_local is not None:
        local = multi_node.data_parallel_size_local
        builder.add("--data-parallel-size-local", local, source="managed multi-node")
        builder.add(
            "--data-parallel-address",
            multi_node.master_addr,
            source="managed multi-node",
        )
        builder.add(
            "--data-parallel-rpc-port",
            multi_node.master_port,
            source="managed multi-node",
        )
        if multi_node.node_rank > 0:
            builder.add(
                "--data-parallel-start-rank",
                multi_node.node_rank * local,
                source="managed multi-node",
            )
            builder.add("--headless", source="managed multi-node")
    elif multi_node is not None:
        builder.add("--nnodes", multi_node.nnodes, source="managed multi-node")
        builder.add("--node-rank", multi_node.node_rank, source="managed multi-node")
        builder.add(
            "--master-addr", multi_node.master_addr, source="managed multi-node"
        )
        builder.add(
            "--master-port", multi_node.master_port, source="managed multi-node"
        )
        if multi_node.node_rank > 0:
            builder.add("--headless", source="managed multi-node")
    builder.add("--model", model_name, source="managed runtime")
    builder.add("--served-model-name", model_name, source="managed runtime")
    builder.add("--namespace", namespace, source="managed runtime")
    builder.add("--discovery-backend", "etcd", source="managed runtime")
    builder.add("--request-plane", "tcp", source="managed runtime")
    builder.add("--event-plane", "nats", source="managed runtime")
    builder.add("--enable-rl", source="managed runtime")
    if worker_role == "aggregated":
        if kv_event_port is not None:
            raise ValueError("Aggregated Dynamo workers do not use a KV-event port.")
    else:
        builder.add(
            "--disaggregation-mode",
            worker_role,
            source="managed disaggregation",
        )
        disaggregation = dynamo_cfg.disaggregation
        if disaggregation is None:
            raise ValueError(
                f"A {worker_role} Dynamo worker requires dynamo_cfg.disaggregation."
            )
        # kv_both is the role Dynamo 1.5 documents for NixlConnector P/D workers,
        # and it is mandatory for prefill workers. A KV pull that fails is
        # recomputed on the decode engine instead of failing the rollout.
        builder.add(
            "--kv-transfer-config",
            {
                "kv_connector": "NixlConnector",
                "kv_role": "kv_both",
                "kv_load_failure_policy": "recompute",
                "kv_connector_extra_config": {
                    "kv_lease_duration": disaggregation.kv_lease_duration_s
                },
            },
            source="managed disaggregation",
        )
        if worker_role == "prefill":
            if kv_event_port is None:
                raise ValueError(
                    "Managed Dynamo prefill workers require a KV-event port."
                )
            builder.add(
                "--kv-events-config",
                {
                    "publisher": "zmq",
                    "topic": "kv-events",
                    "endpoint": f"tcp://*:{kv_event_port}",
                    "enable_kv_cache_events": True,
                },
                source="managed disaggregation",
            )
        elif kv_event_port is not None:
            raise ValueError("Dynamo decode workers do not use a KV-event port.")
    builder.add(
        "--worker-extension-cls",
        DYNAMO_WORKER_EXTENSION_CLS,
        source="managed runtime",
    )
    builder.add("--trust-remote-code", source="managed runtime")
    builder.add("--seed", seed, source="managed runtime")

    role_kwargs: dict[str, Any] = {}
    if worker_role != "aggregated":
        # dynamo_cfg.disaggregation is set: the role branch above required it.
        role_kwargs = dict(
            getattr(dynamo_cfg.disaggregation, f"{worker_role}_vllm_kwargs")
        )
        vllm_cfg = {
            **vllm_cfg,
            **{
                key: role_kwargs.pop(key)
                for key in _ROLE_OVERRIDABLE_CFG_KEYS
                if key in role_kwargs
            },
        }
    for key, flag in DYNAMO_VLLM_FLAGS.items():
        value = vllm_cfg.get(key)
        if value is not None:
            builder.add(flag, value, source=f"vllm_cfg.{key}")

    if int(vllm_cfg.get("data_parallel_size") or 1) > 1:
        builder.add(
            "--data-parallel-size",
            int(vllm_cfg["data_parallel_size"]),
            source="vllm_cfg.data_parallel_size",
        )
    if int(vllm_cfg["expert_parallel_size"]) > 1:
        builder.add(
            "--enable-expert-parallel",
            source="vllm_cfg.expert_parallel_size",
        )

    worker_args = dynamo_cfg.worker_args
    if worker_args.tool_call_parser is not None:
        builder.add(
            "--dyn-tool-call-parser",
            worker_args.tool_call_parser,
            source="dynamo_cfg.worker_args.tool_call_parser",
        )
    if worker_args.reasoning_parser is not None:
        builder.add(
            "--dyn-reasoning-parser",
            worker_args.reasoning_parser,
            source="dynamo_cfg.worker_args.reasoning_parser",
        )
    builder.add(
        "--exclude-tools-when-tool-choice-none",
        worker_args.exclude_tools_when_tool_choice_none,
        source="dynamo_cfg.worker_args.exclude_tools_when_tool_choice_none",
    )
    builder.add(
        "--dyn-enable-structural-tag",
        worker_args.enable_structural_tag,
        source="dynamo_cfg.worker_args.enable_structural_tag",
    )
    builder.add(
        "--dyn-structural-tag-scope",
        worker_args.structural_tag_scope,
        source="dynamo_cfg.worker_args.structural_tag_scope",
    )
    builder.add(
        "--dyn-structural-tag-schema",
        worker_args.structural_tag_schema,
        source="dynamo_cfg.worker_args.structural_tag_schema",
    )
    if worker_args.custom_jinja_template is not None:
        builder.add(
            "--custom-jinja-template",
            worker_args.custom_jinja_template,
            source="dynamo_cfg.worker_args.custom_jinja_template",
        )
    builder.add(
        "--endpoint-types",
        ",".join(worker_args.endpoint_types),
        source="dynamo_cfg.worker_args.endpoint_types",
    )

    engine_kwargs = {**vllm_kwargs, **role_kwargs}
    for key, value in engine_kwargs.items():
        if value is None:
            continue
        # num_speculative_tokens=0 is how a launch script turns speculative decoding
        # off without deleting the block; vLLM rejects it, so drop the block.
        if key == "speculative_config" and value.get("num_speculative_tokens") == 0:
            continue
        flag = _flag_for_key(key)
        normalised = _normalise_flag(flag)
        source = f"vllm_kwargs.{key}"
        if normalised in _MANAGED_FLAGS:
            raise ValueError(f"{source} may not override managed option {normalised}.")
        builder.add(flag, value, source=source)

    builder.add_raw(
        worker_args.extra_cli_args,
        source="dynamo_cfg.worker_args.extra_cli_args",
    )
    return builder.argv


def build_dynamo_frontend_argv(
    *,
    host: str,
    port: int,
    namespace: str,
    dynamo_cfg: DynamoCfg,
) -> list[str]:
    """Build the managed ``dynamo.frontend`` argument list."""
    builder = _ArgvBuilder()
    builder.add("--http-host", host, source="managed runtime")
    builder.add("--http-port", port, source="managed runtime")
    builder.add("--namespace-prefix", namespace, source="managed runtime")
    builder.add("--discovery-backend", "etcd", source="managed runtime")
    builder.add("--request-plane", "tcp", source="managed runtime")
    builder.add("--event-plane", "nats", source="managed runtime")
    builder.add(
        "--router-mode",
        dynamo_cfg.frontend_args.router_mode,
        source="dynamo_cfg.frontend_args.router_mode",
    )
    builder.add_raw(
        dynamo_cfg.frontend_args.extra_cli_args,
        source="dynamo_cfg.frontend_args.extra_cli_args",
    )
    return builder.argv


def build_managed_worker_env(
    *,
    base_env: Mapping[str, str],
    configured_env: Mapping[str, str],
    manager_env: Mapping[str, str],
) -> dict[str, str]:
    """Return a reproducible worker environment with semantic overrides blocked."""
    reserved_env_keys = set(manager_env)
    forbidden = sorted(
        key
        for key in configured_env
        if key.startswith("DYN_") or key in reserved_env_keys
    )
    if forbidden:
        raise ValueError(
            "vllm_cfg.env_vars may not override managed Dynamo settings: "
            + ", ".join(forbidden)
        )

    env = {
        key: value
        for key, value in base_env.items()
        if not key.startswith(("DYN_", "ETCD_", "NATS_"))
        and key not in reserved_env_keys
    }
    env.update(configured_env)
    env.update(manager_env)
    return env


def redact_argv(argv: Sequence[str]) -> list[str]:
    """Redact values following explicit credential options for safe logging."""
    redacted = list(argv)
    for idx, token in enumerate(redacted):
        is_sensitive = token.startswith("--") and (
            _normalise_flag(token.lower()) in _CREDENTIAL_FLAGS
        )
        if is_sensitive and "=" in token:
            redacted[idx] = token.split("=", 1)[0] + "=<redacted>"
        elif is_sensitive and idx + 1 < len(redacted):
            if not redacted[idx + 1].startswith("--"):
                redacted[idx + 1] = "<redacted>"
    return redacted


def redact_environment(env: Mapping[str, str]) -> dict[str, str]:
    """Redact credential-like environment values before logging."""
    sensitive_fragments = ("token", "password", "secret", "api_key", "apikey")
    return {
        key: "<redacted>"
        if any(part in key.lower() for part in sensitive_fragments)
        else value
        for key, value in env.items()
    }
