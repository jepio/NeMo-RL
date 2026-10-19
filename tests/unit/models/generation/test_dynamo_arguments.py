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

import json
import warnings

import pytest
from pydantic import ValidationError

from nemo_rl.models.generation import configure_generation_config
from nemo_rl.models.generation.dynamo.arguments import (
    MultiNodeEngine,
    build_dynamo_frontend_argv,
    build_dynamo_vllm_argv,
    build_managed_worker_env,
    redact_argv,
    redact_environment,
)
from nemo_rl.models.generation.dynamo.config import (
    DynamoCfg,
    DynamoConfig,
    DynamoWorkerArgs,
)


def _config(**overrides) -> dict:
    config = {
        "backend": "dynamo",
        "model_name": "Qwen/Qwen3-0.6B",
        "dynamo_cfg": _dynamo_cfg(),
        "vllm_cfg": {
            "async_engine": True,
            "tensor_parallel_size": 2,
            "pipeline_parallel_size": 1,
            "expert_parallel_size": 2,
            "gpu_memory_utilization": 0.8,
            "max_model_len": 512,
            "precision": "bfloat16",
            "kv_cache_dtype": "auto",
            "load_format": "auto",
            "enforce_eager": False,
            "expose_http_server": False,
            "enable_vllm_metrics_logger": True,
            "vllm_metrics_logger_interval": 1.0,
            "env_vars": None,
        },
        "vllm_kwargs": {},
        "colocated": {"enabled": False},
    }
    config.update(overrides)
    return config


def _dynamo_cfg() -> dict:
    return {
        "engine": "vllm",
        "startup_timeout_s": 600,
        "request_timeout_s": 900,
        "control_timeout_s": 600,
        "metrics_include_prefixes": None,
        "metrics_exclude_prefixes": None,
        "worker_args": {
            "tool_call_parser": None,
            "reasoning_parser": None,
            "exclude_tools_when_tool_choice_none": True,
            "enable_structural_tag": False,
            "structural_tag_scope": "auto",
            "structural_tag_schema": "auto",
            "custom_jinja_template": None,
            "endpoint_types": ["chat", "completions"],
            "extra_cli_args": [],
        },
        "frontend_args": {
            "tokenizer": "default",
            "tokenizer_cache": False,
            "tokenizer_cache_bytes": 50 * 1024 * 1024,
            "router_mode": "kv",
            "extra_cli_args": [],
        },
    }


def _flag_value(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def test_config_derives_world_size_and_rejects_removed_public_fields() -> None:
    validated = DynamoConfig.model_validate(_config())
    assert validated.engine_world_size == 2
    assert validated.dynamo_cfg.disaggregation is None
    for field in ("engine_world_size", "namespace", "dynamo_python", "etcd_port"):
        with pytest.raises(ValidationError, match=field):
            DynamoConfig.model_validate(
                _config(dynamo_cfg={field: 1 if field.endswith("port") else "x"})
            )


def test_disaggregation_config_derives_and_validates_exact_resource_count() -> None:
    config = _config()
    config["dynamo_cfg"]["disaggregation"] = {
        "prefill_workers": 2,
        "decode_workers": 1,
    }
    config["colocated"]["resources"] = {"gpus_per_node": 6, "num_nodes": 1}

    validated = DynamoConfig.model_validate(config)
    assert validated.managed_worker_count == 3
    assert validated.managed_inference_world_size == 6

    config["colocated"]["resources"]["num_nodes"] = None
    DynamoConfig.model_validate(config)

    config["colocated"]["resources"]["gpus_per_node"] = 4
    with pytest.raises(ValidationError, match="configured=4, expected=6"):
        DynamoConfig.model_validate(config)


@pytest.mark.parametrize(
    ("resources", "match"),
    [
        (None, "colocated.resources to be set"),
        ({"gpus_per_node": None, "num_nodes": 1}, "explicit.*gpus_per_node"),
    ],
)
def test_disaggregation_config_requires_inference_resources(resources, match) -> None:
    config = _config()
    config["dynamo_cfg"]["disaggregation"] = {
        "prefill_workers": 1,
        "decode_workers": 1,
    }
    if resources is not None:
        config["colocated"]["resources"] = resources

    with pytest.raises(ValidationError, match=match):
        DynamoConfig.model_validate(config)


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"vllm_cfg": {}}, "nonempty"),
        ({"sglang_cfg": {"foo": 1}}, "sglang_cfg"),
        ({"trtllm_cfg": {"foo": 1}}, "trtllm_cfg"),
        ({"colocated": {"enabled": True}}, "must be false"),
        ({"refit_transport": "mcore"}, "not supported when backend='dynamo'"),
        ({"quant_cfg": "nvfp4"}, "quant_cfg"),
        (
            {
                "vllm_cfg": {
                    **_config()["vllm_cfg"],
                    "speculative_config": {"method": "ngram"},
                },
            },
            "vllm_kwargs.speculative_config",
        ),
        ({"vllm_kwargs": {"worker_extension_cls": "x.Y"}}, "worker_extension_cls"),
    ],
)
def test_config_rejects_unsupported_modes(override, match) -> None:
    if override.get("vllm_cfg") == {}:
        config = _config()
        config["vllm_cfg"] = {}
    else:
        config = _config(**override)
    with pytest.raises(ValidationError, match=match):
        DynamoConfig.model_validate(config)


def test_config_rejects_encoder_cache_reset_and_accepts_reload_api() -> None:
    config = _config()
    config["vllm_cfg"]["reset_encoder_cache_after_weight_update"] = True
    with pytest.raises(ValidationError, match="encoder cache"):
        DynamoConfig.model_validate(config)

    config = _config()
    config["vllm_cfg"]["refit_with_reload_api"] = True
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        DynamoConfig.model_validate(config)


@pytest.mark.parametrize("refit_transport", [None, "nccl_reshard"])
def test_config_accepts_collective_and_nccl_reshard_refit(refit_transport) -> None:
    config = DynamoConfig.model_validate(_config(refit_transport=refit_transport))
    assert config.model_extra["refit_transport"] == refit_transport


@pytest.mark.parametrize(
    ("vllm_cfg", "match"),
    [
        (
            {"tensor_parallel_size": 2, "expert_parallel_size": 3},
            "expert_parallel_size",
        ),
        ({"precision": "float16"}, "precision"),
        ({"precision": "fp8", "kv_cache_dtype": "fp8"}, "kv_cache_dtype"),
        ({"is_mx": True}, "require precision='fp8'"),
        ({"num_last_layers_in_bf16": 2}, "require precision='fp8'"),
    ],
)
def test_config_rejects_unsupported_parallelism_and_precision(vllm_cfg, match) -> None:
    config = _config()
    config["vllm_cfg"].update(vllm_cfg)
    with pytest.raises(ValidationError, match=match):
        DynamoConfig.model_validate(config)


@pytest.mark.parametrize(
    "fp8_fields",
    [{}, {"is_mx": True}, {"num_first_layers_in_bf16": 1, "use_deep_gemm": True}],
)
def test_config_accepts_fp8_generation(fp8_fields) -> None:
    config = _config()
    config["vllm_cfg"].update({"precision": "fp8", **fp8_fields})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        DynamoConfig.model_validate(config)


def test_config_classifies_managed_logprobs_and_runtime_fields() -> None:
    config = _config()
    config["vllm_cfg"].update(
        {
            "logprobs_mode": "processed_logprobs",
            "cap_max_tokens_to_context": False,
            "use_deep_gemm": False,
            "num_first_layers_in_bf16": 0,
            "num_last_layers_in_bf16": 0,
        }
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        DynamoConfig.model_validate(config)
    assert caught == []

    config["vllm_cfg"]["logprobs_mode"] = "raw_logprobs"
    with pytest.raises(ValidationError, match="processed_logprobs"):
        DynamoConfig.model_validate(config)


@pytest.mark.parametrize("field", ["skip_tokenizer_init", "cap_max_tokens_to_context"])
def test_config_warns_only_for_active_unsupported_fields(field) -> None:
    config = _config()
    config["vllm_cfg"][field] = True

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        DynamoConfig.model_validate(config)
    assert [str(warning.message) for warning in caught] == [
        f"policy.generation.vllm_cfg.{field} is ignored by backend='dynamo'"
    ]


@pytest.mark.parametrize(
    "field",
    [
        "data_parallel_size",
        "prefill_context_parallel_size",
        "decode_context_parallel_size",
    ],
)
@pytest.mark.parametrize("source", ["vllm_cfg", "vllm_kwargs"])
def test_config_rejects_parallel_dimensions_outside_tp_pp(field, source) -> None:
    config = _config()
    config[source][field] = 2

    with pytest.raises(ValidationError, match=f"{source}.{field} must be 1"):
        DynamoConfig.model_validate(config)


def test_config_rejects_more_than_32_stop_strings() -> None:
    config = _config(stop_strings=[str(index) for index in range(33)])

    with pytest.raises(ValidationError, match="stop_strings supports at most 32"):
        DynamoConfig.model_validate(config)


def test_config_does_not_limit_stop_token_ids() -> None:
    config = _config(stop_token_ids=list(range(33)))

    assert DynamoConfig.model_validate(config).model_extra["stop_token_ids"] == list(
        range(33)
    )


def test_configure_generation_config_selects_dynamo_load_format() -> None:
    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 1

    training = _config(stop_token_ids=None)
    evaluation = _config(stop_token_ids=None)
    del training["vllm_cfg"]["load_format"]
    del evaluation["vllm_cfg"]["load_format"]

    assert (
        configure_generation_config(training, Tokenizer())["vllm_cfg"]["load_format"]
        == "dummy"
    )
    assert (
        configure_generation_config(evaluation, Tokenizer(), is_eval=True)["vllm_cfg"][
            "load_format"
        ]
        == "auto"
    )


@pytest.mark.parametrize(
    ("speculative_config", "trains_mtp", "load_format"),
    [
        ({"method": "mtp", "num_speculative_tokens": 1}, True, "dummy"),
        (
            {"method": "eagle3", "model": "draft", "num_speculative_tokens": 3},
            False,
            "auto",
        ),
    ],
)
def test_configure_generation_config_applies_speculative_rules_to_dynamo(
    speculative_config, trains_mtp, load_format
) -> None:
    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 1

    config = _config(
        stop_token_ids=None, vllm_kwargs={"speculative_config": speculative_config}
    )
    del config["vllm_cfg"]["load_format"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        configured = configure_generation_config(
            config, Tokenizer(), trains_mtp=trains_mtp
        )

    # A drafter that refit does not feed must start from real weights.
    assert configured["vllm_cfg"]["load_format"] == load_format
    assert configured["_mtp_weights_from_refit"] is trains_mtp
    assert configured["_draft_weights_from_refit"] is False
    DynamoConfig.model_validate(configured)


def test_worker_argv_forwards_speculative_config_unless_disabled() -> None:
    def argv_for(speculative_config):
        return build_dynamo_vllm_argv(
            model_name="model",
            namespace="namespace",
            seed=0,
            vllm_cfg=_config()["vllm_cfg"],
            vllm_kwargs={"speculative_config": speculative_config},
            dynamo_cfg=DynamoCfg.model_validate(_dynamo_cfg()),
        )

    enabled = {"method": "mtp", "num_speculative_tokens": 2}
    assert json.loads(_flag_value(argv_for(enabled), "--speculative-config")) == enabled
    assert "--speculative-config" not in argv_for(
        {"method": "mtp", "num_speculative_tokens": 0}
    )


def test_worker_argv_translates_structured_fields_and_warns_unclassified() -> None:
    config = _dynamo_cfg()
    config["worker_args"].update(
        {"tool_call_parser": "qwen3_coder", "reasoning_parser": "nemotron_nano"}
    )
    cfg = DynamoCfg.model_validate(config)
    generation_config = _config()
    generation_config["vllm_cfg"]["unclassified_field"] = 1
    with pytest.warns(UserWarning, match="unclassified_field"):
        validated = DynamoConfig.model_validate(generation_config)
    vllm_cfg = validated.vllm_cfg.model_dump()
    argv = build_dynamo_vllm_argv(
        model_name="model",
        namespace="nemo-rl-1",
        seed=7,
        vllm_cfg=vllm_cfg,
        vllm_kwargs={"max_num_seqs": 16, "hf_overrides": {"rope_theta": 1e6}},
        dynamo_cfg=cfg,
        worker_role="aggregated",
        kv_event_port=None,
    )

    assert _flag_value(argv, "--model") == "model"
    assert "--weight-transfer-config" not in argv
    assert _flag_value(argv, "--worker-extension-cls") == (
        "nemo_rl.models.generation.vllm.dynamo_worker_extension.DynamoWorkerExtension"
    )
    assert _flag_value(argv, "--dyn-tool-call-parser") == "qwen3_coder"
    assert _flag_value(argv, "--dyn-reasoning-parser") == "nemotron_nano"
    assert _flag_value(argv, "--max-num-seqs") == "16"
    assert _flag_value(argv, "--hf-overrides") == '{"rope_theta":1000000.0}'
    assert "--enable-expert-parallel" in argv


def test_disaggregated_worker_argv_owns_role_and_nixl_configuration() -> None:
    dynamo_cfg = _dynamo_cfg()
    dynamo_cfg["disaggregation"] = {
        "prefill_workers": 1,
        "decode_workers": 1,
        "kv_lease_duration_s": 1200,
    }
    cfg = DynamoCfg.model_validate(dynamo_cfg)
    common = {
        "model_name": "model",
        "namespace": "nemo-rl-1",
        "seed": 7,
        "vllm_cfg": _config()["vllm_cfg"],
        "vllm_kwargs": {},
        "dynamo_cfg": cfg,
    }
    decode_argv = build_dynamo_vllm_argv(
        **common,
        worker_role="decode",
        kv_event_port=None,
    )
    prefill_argv = build_dynamo_vllm_argv(
        **common,
        worker_role="prefill",
        kv_event_port=4201,
    )

    nixl_config = {
        "kv_connector": "NixlConnector",
        "kv_role": "kv_both",
        # A lapsed lease or failed pull must recompute, not fail the rollout.
        "kv_load_failure_policy": "recompute",
        "kv_connector_extra_config": {"kv_lease_duration": 1200},
    }
    assert _flag_value(decode_argv, "--disaggregation-mode") == "decode"
    assert json.loads(_flag_value(decode_argv, "--kv-transfer-config")) == nixl_config
    assert "--kv-events-config" not in decode_argv
    assert _flag_value(prefill_argv, "--disaggregation-mode") == "prefill"
    assert json.loads(_flag_value(prefill_argv, "--kv-transfer-config")) == nixl_config
    assert json.loads(_flag_value(prefill_argv, "--kv-events-config")) == {
        "enable_kv_cache_events": True,
        "endpoint": "tcp://*:4201",
        "publisher": "zmq",
        "topic": "kv-events",
    }


def test_disaggregated_role_kwargs_replace_shared_engine_kwargs_per_role() -> None:
    dynamo_cfg = _dynamo_cfg()
    dynamo_cfg["disaggregation"] = {
        "prefill_workers": 1,
        "decode_workers": 1,
        "prefill_vllm_kwargs": {
            "max_num_batched_tokens": 32768,
            "gpu_memory_utilization": 0.92,
        },
        "decode_vllm_kwargs": {"max_num_seqs": 64},
    }
    cfg = DynamoCfg.model_validate(dynamo_cfg)
    common = {
        "model_name": "model",
        "namespace": "nemo-rl-1",
        "seed": 7,
        "vllm_cfg": _config()["vllm_cfg"],
        "vllm_kwargs": {"max_num_batched_tokens": 16384, "max_num_seqs": 128},
        "dynamo_cfg": cfg,
    }
    prefill_argv = build_dynamo_vllm_argv(
        **common, worker_role="prefill", kv_event_port=4201
    )
    decode_argv = build_dynamo_vllm_argv(
        **common, worker_role="decode", kv_event_port=None
    )

    assert _flag_value(prefill_argv, "--max-num-batched-tokens") == "32768"
    assert _flag_value(prefill_argv, "--max-num-seqs") == "128"
    assert _flag_value(decode_argv, "--max-num-batched-tokens") == "16384"
    assert _flag_value(decode_argv, "--max-num-seqs") == "64"
    assert _flag_value(prefill_argv, "--gpu-memory-utilization") == "0.92"
    assert _flag_value(decode_argv, "--gpu-memory-utilization") == "0.8"


def test_multi_node_engine_argv_marks_the_leader_and_the_headless_nodes() -> None:
    common = {
        "model_name": "model",
        "namespace": "nemo-rl-1",
        "seed": 7,
        "vllm_cfg": _config()["vllm_cfg"],
        "vllm_kwargs": {},
        "dynamo_cfg": DynamoCfg.model_validate(_dynamo_cfg()),
        "worker_role": "aggregated",
        "kv_event_port": None,
    }
    leader = build_dynamo_vllm_argv(
        **common,
        multi_node=MultiNodeEngine(
            nnodes=2, node_rank=0, master_addr="10.0.0.1", master_port=4001
        ),
    )
    follower = build_dynamo_vllm_argv(
        **common,
        multi_node=MultiNodeEngine(
            nnodes=2, node_rank=1, master_addr="10.0.0.1", master_port=4001
        ),
    )
    single = build_dynamo_vllm_argv(**common)

    for argv, rank in ((leader, "0"), (follower, "1")):
        assert _flag_value(argv, "--nnodes") == "2"
        assert _flag_value(argv, "--node-rank") == rank
        assert _flag_value(argv, "--master-addr") == "10.0.0.1"
        assert _flag_value(argv, "--master-port") == "4001"
    assert "--headless" not in leader
    assert "--headless" in follower
    assert "--nnodes" not in single
    assert "--headless" not in single


@pytest.mark.parametrize(
    "flag", ["--nnodes", "--node-rank", "--master-addr", "--master-port", "--headless"]
)
def test_multi_node_flags_are_managed(flag) -> None:
    dynamo_cfg = _dynamo_cfg()
    dynamo_cfg["worker_args"]["extra_cli_args"] = [flag, "1"]
    with pytest.raises(ValueError, match="managed option"):
        build_dynamo_vllm_argv(
            model_name="model",
            namespace="nemo-rl-1",
            seed=7,
            vllm_cfg=_config()["vllm_cfg"],
            vllm_kwargs={},
            dynamo_cfg=DynamoCfg.model_validate(dynamo_cfg),
            worker_role="aggregated",
            kv_event_port=None,
        )


def test_aggregated_worker_ignores_disaggregation_role_kwargs() -> None:
    dynamo_cfg = _dynamo_cfg()
    dynamo_cfg["disaggregation"] = {
        "prefill_workers": 1,
        "decode_workers": 1,
        "prefill_vllm_kwargs": {"max_num_batched_tokens": 32768},
    }
    argv = build_dynamo_vllm_argv(
        model_name="model",
        namespace="nemo-rl-1",
        seed=7,
        vllm_cfg=_config()["vllm_cfg"],
        vllm_kwargs={"max_num_batched_tokens": 16384},
        dynamo_cfg=DynamoCfg.model_validate(dynamo_cfg),
        worker_role="aggregated",
        kv_event_port=None,
    )

    assert _flag_value(argv, "--max-num-batched-tokens") == "16384"


def test_disaggregated_role_kwargs_may_not_override_managed_options() -> None:
    dynamo_cfg = _dynamo_cfg()
    dynamo_cfg["disaggregation"] = {
        "prefill_workers": 1,
        "decode_workers": 1,
        "prefill_vllm_kwargs": {"worker_extension_cls": "x.Y"},
    }
    with pytest.raises(ValueError, match="managed option"):
        build_dynamo_vllm_argv(
            model_name="model",
            namespace="nemo-rl-1",
            seed=7,
            vllm_cfg=_config()["vllm_cfg"],
            vllm_kwargs={},
            dynamo_cfg=DynamoCfg.model_validate(dynamo_cfg),
            worker_role="prefill",
            kv_event_port=4201,
        )


def test_disaggregated_kv_lease_defaults_to_outlast_a_refit() -> None:
    cfg = DynamoCfg.model_validate(
        {
            **_dynamo_cfg(),
            "disaggregation": {"prefill_workers": 1, "decode_workers": 2},
        }
    )
    assert cfg.disaggregation.kv_lease_duration_s == 1800
    assert cfg.disaggregation.kv_lease_duration_s >= 2 * cfg.control_timeout_s


@pytest.mark.parametrize("lease_s", [1199, 600, 30])
def test_disaggregated_kv_lease_must_be_at_least_twice_the_control_timeout(
    lease_s,
) -> None:
    # Decode engines send no heartbeat during a refit pause. A lease that ends
    # first frees KV that decode has not read, and decode then reads wrong KV.
    with pytest.raises(ValueError, match="at least twice control_timeout_s"):
        DynamoCfg.model_validate(
            {
                **_dynamo_cfg(),
                "disaggregation": {
                    "prefill_workers": 1,
                    "decode_workers": 1,
                    "kv_lease_duration_s": lease_s,
                },
            }
        )


def test_disaggregated_kv_lease_of_exactly_twice_the_control_timeout_is_valid() -> None:
    DynamoCfg.model_validate(
        {
            **_dynamo_cfg(),
            "disaggregation": {
                "prefill_workers": 1,
                "decode_workers": 1,
                "kv_lease_duration_s": 1200,
            },
        }
    )


def test_kv_lease_is_not_checked_without_disaggregation() -> None:
    DynamoCfg.model_validate({**_dynamo_cfg(), "control_timeout_s": 1})


@pytest.mark.parametrize("role", ["decode", "prefill"])
def test_disaggregated_role_requires_disaggregation_config(role) -> None:
    with pytest.raises(ValueError, match="requires dynamo_cfg.disaggregation"):
        build_dynamo_vllm_argv(
            model_name="model",
            namespace="namespace",
            seed=0,
            vllm_cfg=_config()["vllm_cfg"],
            vllm_kwargs={},
            dynamo_cfg=DynamoCfg.model_validate(_dynamo_cfg()),
            worker_role=role,
            kv_event_port=4201 if role == "prefill" else None,
        )


def test_worker_argv_rejects_replaced_and_managed_options() -> None:
    generation_config = _config()
    generation_config["vllm_cfg"]["http_server_serving_chat_kwargs"] = {
        "tool_parser": "x"
    }
    with pytest.raises(ValueError, match="worker_args.custom_jinja_template"):
        DynamoConfig.model_validate(generation_config)
    config = _dynamo_cfg()
    config["worker_args"]["extra_cli_args"] = ["--model", "other"]
    with pytest.raises(ValueError, match="--model is set by both"):
        build_dynamo_vllm_argv(
            model_name="model",
            namespace="namespace",
            seed=0,
            vllm_cfg=_config()["vllm_cfg"],
            vllm_kwargs={},
            dynamo_cfg=DynamoCfg.model_validate(config),
            worker_role="aggregated",
            kv_event_port=None,
        )


def test_config_accepts_inherited_unused_sections() -> None:
    config = _config(
        mcore_generation_config={"some_shared_setting": True},
        refit_cfg={"some_shared_setting": True},
    )

    validated = DynamoConfig.model_validate(config)

    assert validated.model_extra["mcore_generation_config"] == {
        "some_shared_setting": True
    }
    assert validated.model_extra["refit_cfg"] == {"some_shared_setting": True}


@pytest.mark.parametrize("value", [True, False])
def test_frontend_rejects_removed_router_reset_states(value) -> None:
    dynamo_cfg = _dynamo_cfg()
    dynamo_cfg["frontend_args"]["router_reset_states"] = value
    with pytest.raises(
        ValueError,
        match=r"frontend_args\.router_reset_states was removed.*--router-reset-states",
    ):
        DynamoCfg.model_validate(dynamo_cfg)


def test_frontend_argv_and_environment_are_runtime_owned() -> None:
    cfg = DynamoCfg.model_validate(_dynamo_cfg())
    argv = build_dynamo_frontend_argv(
        host="0.0.0.0", port=3001, namespace="nemo-rl", dynamo_cfg=cfg
    )
    assert _flag_value(argv, "--router-mode") == "kv"
    assert not any(arg.startswith("--router-reset-states") for arg in argv)
    assert not any(arg.startswith("--no-router-reset-states") for arg in argv)

    env = build_managed_worker_env(
        base_env={
            "DYN_NAMESPACE": "stale",
            "ETCD_ENDPOINTS": "http://stale-etcd:2379",
            "ETCD_USERNAME": "stale-user",
            "NATS_SERVER": "nats://stale-nats:4222",
            "NATS_AUTH_TOKEN": "stale-token",
            "NCCL_DEBUG": "INFO",
        },
        configured_env={"NCCL_IB_DISABLE": "0"},
        manager_env={
            "DYN_NAMESPACE": "owned",
            "DYN_SYSTEM_PORT": "4000",
            "ETCD_ENDPOINTS": "http://managed-etcd:2379",
            "NATS_SERVER": "nats://managed-nats:4222",
        },
    )
    assert env["DYN_NAMESPACE"] == "owned"
    assert env["DYN_SYSTEM_PORT"] == "4000"
    assert env["ETCD_ENDPOINTS"] == "http://managed-etcd:2379"
    assert env["NATS_SERVER"] == "nats://managed-nats:4222"
    assert "ETCD_USERNAME" not in env
    assert "NATS_AUTH_TOKEN" not in env
    with pytest.raises(ValueError, match="VLLM_PORT"):
        build_managed_worker_env(
            base_env={},
            configured_env={"VLLM_PORT": "9999"},
            manager_env={"VLLM_PORT": "7000"},
        )


def test_every_worker_config_field_reaches_argv(monkeypatch) -> None:
    from nemo_rl.models.generation.dynamo import arguments

    sources: set[str] = set()
    original_add = arguments._ArgvBuilder.add

    def record_source(self, flag, value=None, *, source):
        sources.add(source)
        return original_add(self, flag, value, source=source)

    monkeypatch.setattr(arguments._ArgvBuilder, "add", record_source)
    config = _dynamo_cfg()
    config["worker_args"].update(
        {
            "tool_call_parser": "qwen3_coder",
            "reasoning_parser": "nemotron_nano",
            "custom_jinja_template": "template",
        }
    )
    cfg = DynamoCfg.model_validate(config)
    build_dynamo_vllm_argv(
        model_name="model",
        namespace="namespace",
        seed=0,
        vllm_cfg=_config()["vllm_cfg"],
        vllm_kwargs={},
        dynamo_cfg=cfg,
        worker_role="aggregated",
        kv_event_port=None,
    )

    configured_fields = {
        source.rsplit(".", 1)[-1]
        for source in sources
        if source.startswith("dynamo_cfg.worker_args.")
    }
    assert set(DynamoWorkerArgs.model_fields) - {"extra_cli_args"} <= configured_fields


def test_redaction_hides_credentials() -> None:
    assert redact_argv(["worker", "--api-key", "secret"])[2] == "<redacted>"
    assert redact_argv(
        ["worker", "--max-num-batched-tokens", "8192", "--stop-token-ids", "1,2"]
    ) == ["worker", "--max-num-batched-tokens", "8192", "--stop-token-ids", "1,2"]
    assert redact_environment({"HF_TOKEN": "secret", "NCCL_DEBUG": "INFO"}) == {
        "HF_TOKEN": "<redacted>",
        "NCCL_DEBUG": "INFO",
    }


def test_data_parallel_attention_adds_the_data_parallel_size_and_expert_parallelism() -> (
    None
):
    vllm_cfg = {
        **_config()["vllm_cfg"],
        "tensor_parallel_size": 1,
        "data_parallel_size": 4,
        "expert_parallel_size": 4,
    }
    argv = build_dynamo_vllm_argv(
        model_name="model",
        namespace="nemo-rl-1",
        seed=7,
        vllm_cfg=vllm_cfg,
        vllm_kwargs={},
        dynamo_cfg=DynamoCfg.model_validate(_dynamo_cfg()),
        worker_role="aggregated",
        kv_event_port=None,
    )

    assert _flag_value(argv, "--data-parallel-size") == "4"
    assert _flag_value(argv, "--tensor-parallel-size") == "1"
    assert "--enable-expert-parallel" in argv
    single = build_dynamo_vllm_argv(
        model_name="model",
        namespace="nemo-rl-1",
        seed=7,
        vllm_cfg=_config()["vllm_cfg"],
        vllm_kwargs={},
        dynamo_cfg=DynamoCfg.model_validate(_dynamo_cfg()),
        worker_role="aggregated",
        kv_event_port=None,
    )
    assert "--data-parallel-size" not in single


def test_data_parallel_config_derives_world_size_and_checks_expert_parallelism() -> (
    None
):
    config = _config()
    config["vllm_cfg"].update(
        {"tensor_parallel_size": 1, "data_parallel_size": 4, "expert_parallel_size": 4}
    )
    assert DynamoConfig.model_validate(config).engine_world_size == 4

    config["vllm_cfg"]["expert_parallel_size"] = 2
    with pytest.raises(
        ValidationError, match="tensor_parallel_size \\* data_parallel_size"
    ):
        DynamoConfig.model_validate(config)
    config["vllm_cfg"]["expert_parallel_size"] = 1
    with pytest.raises(ValidationError, match="data_parallel_size > 1"):
        DynamoConfig.model_validate(config)


def test_data_parallel_size_is_managed_in_vllm_kwargs() -> None:
    with pytest.raises(ValueError, match="managed option"):
        build_dynamo_vllm_argv(
            model_name="model",
            namespace="nemo-rl-1",
            seed=7,
            vllm_cfg=_config()["vllm_cfg"],
            vllm_kwargs={"data_parallel_size": 2},
            dynamo_cfg=DynamoCfg.model_validate(_dynamo_cfg()),
            worker_role="aggregated",
            kv_event_port=None,
        )


def test_multi_node_data_parallel_argv_uses_the_data_parallel_launch() -> None:
    vllm_cfg = {
        **_config()["vllm_cfg"],
        "tensor_parallel_size": 1,
        "data_parallel_size": 8,
        "expert_parallel_size": 8,
    }
    common = {
        "model_name": "model",
        "namespace": "nemo-rl-1",
        "seed": 7,
        "vllm_cfg": vllm_cfg,
        "vllm_kwargs": {},
        "dynamo_cfg": DynamoCfg.model_validate(_dynamo_cfg()),
        "worker_role": "aggregated",
        "kv_event_port": None,
    }
    leader = build_dynamo_vllm_argv(
        **common,
        multi_node=MultiNodeEngine(
            nnodes=2,
            node_rank=0,
            master_addr="10.0.0.1",
            master_port=4001,
            data_parallel_size_local=4,
        ),
    )
    follower = build_dynamo_vllm_argv(
        **common,
        multi_node=MultiNodeEngine(
            nnodes=2,
            node_rank=1,
            master_addr="10.0.0.1",
            master_port=4001,
            data_parallel_size_local=4,
        ),
    )

    for argv in (leader, follower):
        assert _flag_value(argv, "--data-parallel-size") == "8"
        assert _flag_value(argv, "--data-parallel-size-local") == "4"
        assert _flag_value(argv, "--data-parallel-address") == "10.0.0.1"
        assert _flag_value(argv, "--data-parallel-rpc-port") == "4001"
        assert "--nnodes" not in argv
        assert "--node-rank" not in argv
    assert "--data-parallel-start-rank" not in leader
    assert "--headless" not in leader
    assert _flag_value(follower, "--data-parallel-start-rank") == "4"
    assert "--headless" in follower
