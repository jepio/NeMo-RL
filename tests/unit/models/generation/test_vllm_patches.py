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

"""Guards for vLLM source patches and scoped runtime workarounds.

The two port patches ship their own suites. This module covers the other
source-sensitive compatibility patches:

* ``_patch_vllm_tool_parser_namespace_tool`` is the most load-bearing patch in
  the repo -- it is the only thing that makes vLLM 0.25.1 importable against
  the pinned ``openai==2.6.1``. If upstream reorders that import block the
  patch logs a warning and returns, and every engine then dies on
  ``import vllm.tool_parsers``. So the anchor needs pinning.
* ``_patch_vllm_glm_decoder_sequence_parallel_moe`` restores the vLLM 0.24
  decoder boundary for GLM-5.1/5.2 while leaving MoE-local SP enabled.
* the ``VLLM_RAY_EXTRA_ENV_VARS_TO_COPY`` merge replaced the old
  ``ADDITIONAL_ENV_VARS`` file patch and is what now carries
  ``RAY_ENABLE_UV_RUN_RUNTIME_ENV`` and every user ``extra_env_vars`` to the
  Ray workers. Being additive rather than clobbering is the whole point of the
  rewrite, and it is pure string handling, so it is cheap to pin.
* ``modelopt_moe_amax_aliases`` adapts nested ModelOpt buffers to vLLM's
  MoE refit loader. Its lifecycle and installed-loader compatibility are
  checked here, alongside the source patches.
"""

import ast
import logging
import os
import sys
import types
from types import SimpleNamespace

import pytest
import torch

from nemo_rl.models.generation.vllm import patches
from nemo_rl.models.generation.vllm.config import (
    VLLM_NEMOTRON_H_FP32_LM_HEAD_ENV_VAR,
    vllm_nemotron_h_fp32_lm_head_enabled,
)
from tests.unit.models.generation.vllm_patch_source_utils import (
    patch_snippets,
    write_unpatched_copy,
)

_TOOL_PARSER_SOURCE = "tool_parsers/utils.py"
_PATCH_FN = "_patch_vllm_tool_parser_namespace_tool"
_MARKER = "except ImportError:  # openai < 2.25.0 predates namespace tools"
_PAUSE_SOURCE = "v1/core/sched/scheduler.py"
_PAUSE_PATCH_FN = "_patch_vllm_pause_with_held_kv"
_PAUSE_MARKER = "NeMo RL: a PAUSED_ALL scheduler does not wait for connector-held"
_RESET_PATCH_FN = "_patch_vllm_reset_cache_with_held_kv"
_RESET_MARKER = "NeMo RL: drop every cached prefix hash while some blocks stay in use"
_STALE_PATCH_FN = "_patch_vllm_stale_remote_kv_after_reset"
_STALE_MARKER = "NeMo RL: do not leave remote KV from the old weights cacheable"
_RADIO_SOURCE = "model_executor/models/radio.py"
_RADIO_PATCH_FN = "_patch_vllm_radio_layerscale_loader"
_RADIO_MARKER = "initializer_factor = self.config.initializer_factor"
_GLM_DSA_SOURCE = "model_executor/models/deepseek_v2.py"
_GLM_DSA_PATCH_FN = "_patch_vllm_glm_decoder_sequence_parallel_moe"
_GLM_DSA_MARKER = 'getattr(config, "model_type", None) != "glm_moe_dsa"'
_NEMOTRON_H_SOURCE = """import torch
from torch import nn


def maybe_prefix(prefix, name):
    return f"{prefix}.{name}"


class LogitsProcessor:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size

    def __call__(self, lm_head, hidden_states):
        return lm_head.quant_method.apply(lm_head, hidden_states)


class QuantMethod:
    def __init__(self):
        self.seen_dtypes = []

    def apply(self, lm_head, hidden_states, bias=None):
        self.seen_dtypes.append(
            (
                hidden_states.dtype,
                lm_head.weight.dtype,
                None if bias is None else bias.dtype,
            )
        )
        logits = hidden_states @ lm_head.weight.t()
        if bias is not None:
            logits = logits + bias
        return logits


class ParallelLMHead(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_size,
        params_dtype=None,
        quant_config=None,
        prefix="",
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.params_dtype = params_dtype
        self.quant_config = quant_config
        self.prefix = prefix
        self.weight = nn.Parameter(
            torch.ones(vocab_size, hidden_size, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.bias = None
        self.quant_method = QuantMethod()

    def forward(self, input_):
        del input_
        raise RuntimeError("LMHead's weights should be used in the sampler.")


class NemotronHForCausalLM:
    def __init__(self, config, prefix):
        self.quant_config = object()
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def compute_logits(self, hidden_states):
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits
"""
_MOE_SOURCE = "model_executor/layers/fused_moe/runner/moe_runner.py"
_MOE_PATCH_FN = "_patch_vllm_moe_routed_experts_capture"
_MOE_MARKER = "NeMo-RL patch (routed-experts capture for router replay)"
_CAPTURER_SOURCE = "model_executor/layers/fused_moe/routed_experts_capturer.py"
_CAPTURER_PATCH_FN = "_patch_vllm_routed_experts_capture_router_fallback"
_CAPTURER_MARKER = (
    "NeMo-RL patch (router fallback for monolithic routed-experts capture)"
)


@pytest.fixture
def modelopt_moe_model() -> torch.nn.Module:
    model = torch.nn.Module()
    model.experts = torch.nn.Module()
    for name in ("w13_input_quantizer", "w2_input_quantizer"):
        quantizer = torch.nn.Module()
        quantizer.register_buffer("_amax", torch.tensor(-1.0))
        model.experts.add_module(name, quantizer)
    model.in_proj = torch.nn.Linear(1, 1)
    model.in_proj.input_quantizer = torch.nn.Module()
    model.in_proj.input_quantizer.register_buffer("_amax", torch.tensor(-1.0))
    return model


def test_modelopt_moe_amax_aliases_preserve_buffer_identity_and_registration(
    modelopt_moe_model: torch.nn.Module,
) -> None:
    model = modelopt_moe_model
    buffers_before = list(model.named_buffers())
    parameters_before = list(model.named_parameters())
    state_before = {name: value.clone() for name, value in model.state_dict().items()}

    with patches.modelopt_moe_amax_aliases(model):
        for name in ("w13_input_quantizer", "w2_input_quantizer"):
            assert (
                getattr(model.experts, f"{name}._amax")
                is getattr(model.experts, name)._amax
            )
        assert list(model.named_buffers()) == buffers_before
        assert list(model.named_parameters()) == parameters_before
        torch.testing.assert_close(model.state_dict(), state_before)
        assert not hasattr(model.in_proj, "input_quantizer._amax")

    assert not hasattr(model.experts, "w13_input_quantizer._amax")
    assert not hasattr(model.experts, "w2_input_quantizer._amax")
    torch.testing.assert_close(model.state_dict(), state_before)


def test_modelopt_moe_amax_aliases_support_nested_and_repeated_use(
    modelopt_moe_model: torch.nn.Module,
) -> None:
    model = modelopt_moe_model
    for _ in range(2):
        with patches.modelopt_moe_amax_aliases(model):
            with patches.modelopt_moe_amax_aliases(model):
                assert getattr(model.experts, "w13_input_quantizer._amax") is (
                    model.experts.w13_input_quantizer._amax
                )
            assert hasattr(model.experts, "w13_input_quantizer._amax")
        assert not hasattr(model.experts, "w13_input_quantizer._amax")
        assert not hasattr(model.experts, "w2_input_quantizer._amax")


def test_modelopt_moe_amax_aliases_preserve_existing_attributes(
    modelopt_moe_model: torch.nn.Module,
) -> None:
    model = modelopt_moe_model
    existing = torch.tensor(123.0)
    setattr(model.experts, "w13_input_quantizer._amax", existing)
    with patches.modelopt_moe_amax_aliases(model):
        assert getattr(model.experts, "w13_input_quantizer._amax") is existing
        assert hasattr(model.experts, "w2_input_quantizer._amax")
    assert getattr(model.experts, "w13_input_quantizer._amax") is existing
    assert not hasattr(model.experts, "w2_input_quantizer._amax")


@pytest.mark.parametrize("during_setup", [False, True])
def test_modelopt_moe_amax_aliases_clean_up_on_error(
    modelopt_moe_model: torch.nn.Module,
    monkeypatch: pytest.MonkeyPatch,
    during_setup: bool,
) -> None:
    model = modelopt_moe_model

    def fail_buffer_scan(*, recurse: bool = True) -> None:
        # The first quantizer's alias must already exist when the second fails.
        assert hasattr(model.experts, "w13_input_quantizer._amax")
        raise RuntimeError("quantizer scan failed")

    if during_setup:
        monkeypatch.setattr(
            model.experts.w2_input_quantizer, "named_buffers", fail_buffer_scan
        )
    expected = "quantizer scan failed" if during_setup else "refit failed"
    with pytest.raises(RuntimeError, match=expected):
        with patches.modelopt_moe_amax_aliases(model):
            raise RuntimeError("refit failed")
    assert not hasattr(model.experts, "w13_input_quantizer._amax")
    assert not hasattr(model.experts, "w2_input_quantizer._amax")


@pytest.mark.vllm
def test_modelopt_moe_amax_aliases_satisfy_installed_vllm_loader(
    modelopt_moe_model: torch.nn.Module,
) -> None:
    # Keep the optional vLLM import inside the test selected by its marker.
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

    class LoaderFixture(torch.nn.Module):
        """CPU state for the installed loader and its real expert mapping."""

        load_weights = RoutedExperts.load_weights
        get_expert_mapping = RoutedExperts.get_expert_mapping
        build_expert_params_mapping = staticmethod(
            RoutedExperts.build_expert_params_mapping
        )

        def __init__(self) -> None:
            super().__init__()
            self.layer_name = "model.layers.1.mixer.experts"
            self.moe_config = SimpleNamespace(
                hidden_dim_unpadded=1, num_experts=2, num_logical_experts=2
            )
            self.expert_map_manager = SimpleNamespace(num_fused_shared_experts=0)
            self.ckpt_gate_proj_name = "up_proj"
            self.ckpt_down_proj_name = "down_proj"
            self.ckpt_up_proj_name = ""
            self.lora_base_layer_prefix = ""

    def load_amax(
        param: torch.Tensor, loaded_weight: torch.Tensor, **kwargs: object
    ) -> bool:
        param.copy_(torch.maximum(param, loaded_weight))
        return True

    model = modelopt_moe_model
    experts = LoaderFixture()
    for name, quantizer in model.experts.named_children():
        experts.add_module(name, quantizer)
        quantizer._amax.weight_loader = load_amax
    model.experts = experts
    weights = [
        (f"{expert}.{projection}.input_quantizer._amax", torch.tensor(value))
        for expert, projection, value in (
            (0, "up_proj", 2.0),
            (1, "up_proj", 6.0),
            (0, "down_proj", 1.0),
            (1, "down_proj", 0.25),
        )
    ]

    with pytest.raises(AttributeError, match=r"w13_input_quantizer\._amax"):
        list(experts.load_weights(weights))
    with patches.modelopt_moe_amax_aliases(model):
        loaded = list(experts.load_weights(weights))
        assert loaded == [
            "w13_input_quantizer._amax",
            "w13_input_quantizer._amax",
            "w2_input_quantizer._amax",
            "w2_input_quantizer._amax",
        ]
    torch.testing.assert_close(experts.w13_input_quantizer._amax, torch.tensor(6.0))
    torch.testing.assert_close(experts.w2_input_quantizer._amax, torch.tensor(1.0))
    assert not hasattr(experts, "w13_input_quantizer._amax")
    assert not hasattr(experts, "w2_input_quantizer._amax")


@pytest.fixture
def patched_tool_parser_source(tmp_path, monkeypatch):
    """The installed tool_parsers/utils.py, unpatched then patched in tmp."""
    copied = write_unpatched_copy(_TOOL_PARSER_SOURCE, _PATCH_FN, tmp_path / "utils.py")
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _relative: str(copied))
    patches._patch_vllm_tool_parser_namespace_tool(logging.getLogger(__name__))
    return copied


@pytest.fixture
def patched_radio_source(tmp_path, monkeypatch):
    """The installed vLLM RADIO loader, unpatched then patched in tmp."""
    copied = write_unpatched_copy(_RADIO_SOURCE, _RADIO_PATCH_FN, tmp_path / "radio.py")
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _relative: str(copied))
    patches._patch_vllm_radio_layerscale_loader(logging.getLogger(__name__))
    return copied


@pytest.fixture
def patched_glm_dsa_source(tmp_path, monkeypatch):
    """The installed GLM/DeepSeek model source, unpatched then patched in tmp."""
    copied = write_unpatched_copy(
        _GLM_DSA_SOURCE, _GLM_DSA_PATCH_FN, tmp_path / "deepseek_v2.py"
    )
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _relative: str(copied))
    patches._patch_vllm_glm_decoder_sequence_parallel_moe(logging.getLogger(__name__))
    return copied


@pytest.fixture
def patched_nemotron_h_source(tmp_path, monkeypatch):
    source = tmp_path / "nemotron_h.py"
    source.write_text(_NEMOTRON_H_SOURCE)
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _relative: str(source))
    patches._patch_vllm_nemotron_h_fp32_lm_head(logging.getLogger(__name__))
    return source


@pytest.fixture
def patched_moe_source(tmp_path, monkeypatch):
    """The installed monolithic MoE runner, unpatched then patched in tmp."""
    copied = write_unpatched_copy(
        _MOE_SOURCE, _MOE_PATCH_FN, tmp_path / "moe_runner.py"
    )
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _relative: str(copied))
    assert patches._patch_vllm_moe_routed_experts_capture(
        logging.getLogger(__name__), required=True
    )
    return copied


@pytest.mark.vllm
def test_namespace_tool_patch_anchor_still_matches_installed_vllm(
    patched_tool_parser_source,
):
    """A source edit becomes a silent no-op if upstream reorders the import."""
    content = patched_tool_parser_source.read_text()
    assert _MARKER in content, (
        "the NamespaceTool compat patch did not apply to the installed vLLM; "
        "its anchor import block has probably changed upstream. Every vLLM "
        "engine will fail to import tool_parsers against the pinned openai."
    )
    ast.parse(content)  # the edit must leave valid Python


@pytest.mark.vllm
def test_namespace_tool_patch_is_idempotent(patched_tool_parser_source, monkeypatch):
    """Every worker on a node runs the patch against the same file."""
    before = patched_tool_parser_source.read_text()
    monkeypatch.setattr(
        patches, "_get_vllm_file", lambda _relative: str(patched_tool_parser_source)
    )
    patches._patch_vllm_tool_parser_namespace_tool(logging.getLogger(__name__))
    assert patched_tool_parser_source.read_text() == before


@pytest.mark.vllm
def test_namespace_tool_stub_never_matches(patched_tool_parser_source):
    """The stub must be a plain class, so isinstance() is always False.

    All upstream uses are ``isinstance(tool, NamespaceTool)`` guarding a
    namespace-tools branch, so degrading to "no namespace tools" is correct for
    a client that cannot construct them -- but only if nothing can be an
    instance of the stub.
    """
    namespace: dict = {}
    tree = ast.parse(patched_tool_parser_source.read_text())
    stub = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "NamespaceTool"
    )
    exec(compile(ast.Module(body=[stub], type_ignores=[]), "<stub>", "exec"), namespace)
    stub_cls = namespace["NamespaceTool"]
    for value in ({}, "tool", 0, None, object()):
        assert not isinstance(value, stub_cls)


@pytest.mark.vllm
def test_radio_layerscale_patch_anchor_still_matches_installed_vllm(
    patched_radio_source,
):
    """Pin the vLLM 0.25.1 RADIO loader shape used by the source patch."""
    content = patched_radio_source.read_text()
    assert _RADIO_MARKER in content
    assert "Skip layer-scale entries that vLLM doesn't use" not in content
    ast.parse(content)


@pytest.mark.vllm
def test_radio_layerscale_patch_loads_explicit_and_initializes_folded_weights(
    patched_radio_source,
):
    content = patched_radio_source.read_text()
    assert 'vllm_key = f"model.encoder.layers.{layer_idx}.{suffix}"' in content
    assert 'name.endswith((".ls1", ".ls2"))' in content
    assert "param.data.fill_(initializer_factor)" in content
    assert "loaded_params.add(name)" in content


@pytest.mark.vllm
def test_radio_layerscale_patch_is_idempotent(patched_radio_source, monkeypatch):
    before = patched_radio_source.read_text()
    monkeypatch.setattr(
        patches, "_get_vllm_file", lambda _relative: str(patched_radio_source)
    )

    patches._patch_vllm_radio_layerscale_loader(logging.getLogger(__name__))

    assert patched_radio_source.read_text() == before


def test_radio_layerscale_patch_warns_on_unknown_source(monkeypatch, tmp_path, caplog):
    radio_source = tmp_path / "radio.py"
    radio_source.write_text("class RadioModel:\n    pass\n")
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _relative: str(radio_source))

    with caplog.at_level(logging.WARNING):
        patches._patch_vllm_radio_layerscale_loader(logging.getLogger(__name__))

    assert radio_source.read_text() == "class RadioModel:\n    pass\n"
    assert "vLLM 0.25.1 source shape was not found" in caplog.text


@pytest.mark.vllm
def test_glm_decoder_sp_moe_patch_anchor_still_matches_installed_vllm(
    patched_glm_dsa_source,
):
    """Pin the vLLM 0.25.1 decoder-level SP-MoE source shape."""
    content = patched_glm_dsa_source.read_text()
    assert _GLM_DSA_MARKER in content
    ast.parse(content)


@pytest.mark.vllm
def test_moe_routed_experts_patch_anchor_still_matches_installed_vllm(
    patched_moe_source,
):
    content = patched_moe_source.read_text()
    assert _MOE_MARKER in content
    assert "self.router.select_experts(" in content
    assert 'getattr(self.router, "capture_fn", None)' in content
    ast.parse(content)


@pytest.mark.vllm
def test_glm_decoder_sp_moe_patch_is_idempotent(patched_glm_dsa_source, monkeypatch):
    before = patched_glm_dsa_source.read_text()
    monkeypatch.setattr(
        patches, "_get_vllm_file", lambda _relative: str(patched_glm_dsa_source)
    )

    patches._patch_vllm_glm_decoder_sequence_parallel_moe(logging.getLogger(__name__))

    assert patched_glm_dsa_source.read_text() == before


def test_glm_decoder_sp_moe_patch_warns_on_unknown_source(
    monkeypatch, tmp_path, caplog
):
    model_source = tmp_path / "deepseek_v2.py"
    model_source.write_text("class DeepseekV2DecoderLayer:\n    pass\n")
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _relative: str(model_source))

    with caplog.at_level(logging.WARNING):
        patches._patch_vllm_glm_decoder_sequence_parallel_moe(
            logging.getLogger(__name__)
        )

    assert model_source.read_text() == "class DeepseekV2DecoderLayer:\n    pass\n"
    assert "vLLM 0.25.1 source shape was not found" in caplog.text


@pytest.mark.vllm
def test_moe_routed_experts_patch_is_idempotent(patched_moe_source, monkeypatch):
    before = patched_moe_source.read_text()
    monkeypatch.setattr(
        patches, "_get_vllm_file", lambda _relative: str(patched_moe_source)
    )

    assert patches._patch_vllm_moe_routed_experts_capture(
        logging.getLogger(__name__), required=True
    )
    assert patched_moe_source.read_text() == before


def test_moe_routed_experts_patch_fails_closed_when_required(monkeypatch, tmp_path):
    moe_source = tmp_path / "moe_runner.py"
    moe_source.write_text("class MoERunner:\n    pass\n")
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _relative: str(moe_source))

    with pytest.raises(RuntimeError, match="expected code snippet not found"):
        patches._patch_vllm_moe_routed_experts_capture(
            logging.getLogger(__name__), required=True
        )


@pytest.fixture
def patched_capturer_source(tmp_path, monkeypatch):
    """The installed routed-experts binder, unpatched then patched in tmp."""
    copied = write_unpatched_copy(
        _CAPTURER_SOURCE, _CAPTURER_PATCH_FN, tmp_path / "routed_experts_capturer.py"
    )
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _relative: str(copied))
    assert patches._patch_vllm_routed_experts_capture_router_fallback(
        logging.getLogger(__name__), required=True
    )
    return copied


@pytest.mark.vllm
def test_capture_router_fallback_patch_anchor_still_matches_installed_vllm(
    patched_capturer_source,
):
    content = patched_capturer_source.read_text()
    assert _CAPTURER_MARKER in content
    # The patched monolithic block runs from the marker to the original raise.
    monolithic_branch = content.split(_CAPTURER_MARKER, 1)[1]
    monolithic_branch = monolithic_branch.split("not supported with monolithic", 1)[0]
    # In-kernel capture still wins when the kernel supports it ...
    assert "fused_experts.set_capture_fn(capture_fn)" in monolithic_branch
    # ... and a kernel without it now falls back to the router instead of raising.
    assert "module.router.set_capture_fn(capture_fn)" in monolithic_branch
    assert monolithic_branch.index(
        "fused_experts.set_capture_fn"
    ) < monolithic_branch.index("module.router.set_capture_fn")
    ast.parse(content)


@pytest.mark.vllm
def test_capture_router_fallback_patch_is_idempotent(
    patched_capturer_source, monkeypatch
):
    before = patched_capturer_source.read_text()
    monkeypatch.setattr(
        patches, "_get_vllm_file", lambda _relative: str(patched_capturer_source)
    )

    assert patches._patch_vllm_routed_experts_capture_router_fallback(
        logging.getLogger(__name__), required=True
    )
    assert patched_capturer_source.read_text() == before


def test_capture_router_fallback_patch_fails_closed_when_required(
    monkeypatch, tmp_path
):
    source = tmp_path / "routed_experts_capturer.py"
    source.write_text("def bind_routed_experts_capturer(model, capturer):\n    pass\n")
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _relative: str(source))

    with pytest.raises(RuntimeError, match="expected code snippet not found"):
        patches._patch_vllm_routed_experts_capture_router_fallback(
            logging.getLogger(__name__), required=True
        )
    assert source.read_text().startswith("def bind_routed_experts_capturer")


def test_capture_router_fallback_patch_binds_router_for_unsupported_kernel(
    monkeypatch, tmp_path
):
    """Execute the patched binder body against stand-ins for both kernel kinds."""

    old_snippet, _new_snippet = patch_snippets(_CAPTURER_PATCH_FN)
    source = tmp_path / "routed_experts_capturer.py"
    source.write_text(
        "def bind(module, quant_method, fused_experts, capture_fn, "
        "FusedMoEExpertsMonolithic, BaseRouter):\n"
        "    num_bound = 0\n"
        "    if True:\n" + old_snippet + "        return num_bound\n"
    )
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _relative: str(source))
    assert patches._patch_vllm_routed_experts_capture_router_fallback(
        logging.getLogger(__name__), required=True
    )
    namespace: dict = {}
    exec(compile(source.read_text(), str(source), "exec"), namespace)

    class Monolithic:
        def __init__(self, supports):
            self.supports = supports
            self.bound = None

        def supports_routing_replay_capture(self):
            return self.supports

        def set_capture_fn(self, fn):
            self.bound = fn

    class Router:
        def __init__(self):
            self.bound = None

        def set_capture_fn(self, fn):
            self.bound = fn

    capture_fn = object()
    quant_method = SimpleNamespace(is_monolithic=True)

    kernel, router = Monolithic(True), Router()
    module = SimpleNamespace(router=router)
    assert (
        namespace["bind"](module, quant_method, kernel, capture_fn, Monolithic, Router)
        == 1
    )
    assert kernel.bound is capture_fn and router.bound is None

    kernel, router = Monolithic(False), Router()
    module = SimpleNamespace(router=router)
    assert (
        namespace["bind"](module, quant_method, kernel, capture_fn, Monolithic, Router)
        == 1
    )
    assert kernel.bound is None and router.bound is capture_fn

    kernel = Monolithic(False)
    module = SimpleNamespace(router=object())
    with pytest.raises(ValueError, match="not supported with monolithic"):
        namespace["bind"](module, quant_method, kernel, capture_fn, Monolithic, Router)


@pytest.mark.parametrize(
    ("vllm_cfg", "expected"),
    [
        ({}, False),
        ({"fp32_lm_head": False}, False),
        ({"fp32_lm_head": True}, True),
        ({"env_vars": {VLLM_NEMOTRON_H_FP32_LM_HEAD_ENV_VAR: "1"}}, False),
        ({"env_vars": {VLLM_NEMOTRON_H_FP32_LM_HEAD_ENV_VAR: "0"}}, False),
    ],
)
def test_vllm_nemotron_h_fp32_lm_head_enabled(vllm_cfg, expected):
    assert vllm_nemotron_h_fp32_lm_head_enabled(vllm_cfg) is expected


@pytest.mark.parametrize("env_value", [None, "0", "1"])
def test_nemotron_h_fp32_lm_head_patch_is_env_gated(
    patched_nemotron_h_source, monkeypatch, env_value
):
    if env_value is None:
        monkeypatch.delenv(VLLM_NEMOTRON_H_FP32_LM_HEAD_ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(VLLM_NEMOTRON_H_FP32_LM_HEAD_ENV_VAR, env_value)

    namespace = {}
    source = patched_nemotron_h_source.read_text()
    exec(compile(source, str(patched_nemotron_h_source), "exec"), namespace)
    config = types.SimpleNamespace(vocab_size=16, hidden_size=8)
    model = namespace["NemotronHForCausalLM"](config, "model")
    hidden_states = torch.ones(2, 8, dtype=torch.bfloat16)

    logits = model.compute_logits(hidden_states)

    if env_value == "1":
        assert model._nrl_fp32_lm_head is True
        assert model.lm_head.params_dtype is None
        assert model.lm_head.quant_config is model.quant_config
        assert model.lm_head.weight.dtype is torch.bfloat16
        assert logits.dtype is torch.float32
        assert model.lm_head(hidden_states).dtype is torch.float32
        assert model.lm_head.quant_method.seen_dtypes == []
    else:
        assert model._nrl_fp32_lm_head is False
        assert model.lm_head.params_dtype is None
        assert model.lm_head.quant_config is model.quant_config
        assert model.lm_head.weight.dtype is torch.bfloat16
        assert logits.dtype is torch.bfloat16
        assert model.lm_head.quant_method.seen_dtypes == [
            (torch.bfloat16, torch.bfloat16, None)
        ]

    assert "deepcopy" not in source
    assert "params_dtype=torch.float32" not in source
    assert "NemotronH vLLM lm_head.forward casts " in source
    assert "input and weight to fp32" in source
    assert "torch.matmul(" in source
    ast.parse(source)


def test_nemotron_h_fp32_lm_head_patch_is_idempotent(
    patched_nemotron_h_source, monkeypatch
):
    before = patched_nemotron_h_source.read_text()
    monkeypatch.setattr(
        patches, "_get_vllm_file", lambda _relative: str(patched_nemotron_h_source)
    )

    patches._patch_vllm_nemotron_h_fp32_lm_head(logging.getLogger(__name__))

    assert patched_nemotron_h_source.read_text() == before


def test_nemotron_h_fp32_lm_head_patch_warns_on_unknown_source(
    tmp_path, monkeypatch, caplog
):
    source = tmp_path / "nemotron_h.py"
    source.write_text("class NemotronHForCausalLM:\n    pass\n")
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _relative: str(source))

    with caplog.at_level(logging.WARNING):
        applied = patches._patch_vllm_nemotron_h_fp32_lm_head(
            logging.getLogger(__name__)
        )

    assert applied is False
    assert source.read_text() == "class NemotronHForCausalLM:\n    pass\n"
    assert "NemotronH fp32 LM head import anchor not found exactly once" in caplog.text


@pytest.mark.vllm
def test_nemotron_h_fp32_lm_head_patch_anchor_still_matches_installed_vllm(
    tmp_path, monkeypatch
):
    """Pin the vLLM 0.25.1 Nemotron-H source shape used by the patch."""
    copied = tmp_path / "nemotron_h.py"
    with open(patches._get_vllm_file("model_executor/models/nemotron_h.py")) as f:
        copied.write_text(f.read())
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _relative: str(copied))

    applied = patches._patch_vllm_nemotron_h_fp32_lm_head(logging.getLogger(__name__))

    assert applied is True
    content = copied.read_text()
    assert "self._nrl_fp32_lm_head = (" in content
    assert "def _nrl_fp32_lm_head_forward(" in content
    assert content.index("import os\n") < content.index("import torch\n")
    ast.parse(content)


def _install_fake_vllm_modules(monkeypatch):
    vllm_module = types.ModuleType("vllm")
    envs_module = types.ModuleType("vllm.envs")
    envs_module.VLLM_USE_RAY_V2_EXECUTOR_BACKEND = True
    logger_module = types.ModuleType("vllm.logger")
    logger_module.init_logger = lambda name: logging.getLogger(name)
    vllm_module.envs = envs_module
    vllm_module.logger = logger_module
    monkeypatch.setitem(sys.modules, "vllm", vllm_module)
    monkeypatch.setitem(sys.modules, "vllm.envs", envs_module)
    monkeypatch.setitem(sys.modules, "vllm.logger", logger_module)


def _stub_non_fp32_vllm_patches(monkeypatch, captured_extra_env_vars):
    monkeypatch.setattr(
        patches,
        "_patch_vllm_init_workers_ray",
        lambda _py, extra: captured_extra_env_vars.append(extra) or False,
    )
    for patch_name in (
        "_patch_vllm_llama_eagle3_own_lm_head",
        "_patch_vllm_tool_parser_namespace_tool",
        "_patch_vllm_ray_executor_v2_tcpstore_port",
        "_patch_vllm_shm_broadcast_bind_retry",
        "_patch_vllm_radio_layerscale_loader",
        "_patch_vllm_glm_decoder_sequence_parallel_moe",
    ):
        monkeypatch.setattr(patches, patch_name, lambda _logger: None)
    monkeypatch.setattr(
        patches,
        "_patch_vllm_moe_routed_experts_capture",
        lambda _logger, *, required=False: True,
    )
    monkeypatch.setattr(
        patches,
        "_patch_vllm_routed_experts_capture_router_fallback",
        lambda _logger, *, required=False: True,
    )


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("require_capture", [False, True])
def test_apply_vllm_patches_gates_nemotron_h_fp32_lm_head(
    monkeypatch, enabled, require_capture: bool
):
    _install_fake_vllm_modules(monkeypatch)
    monkeypatch.delenv(patches.VLLM_NEMOTRON_H_FP32_LM_HEAD_ENV_VAR, raising=False)
    captured_extra_env_vars = []
    fp32_patch_calls = []
    capture_requirements = []
    _stub_non_fp32_vllm_patches(monkeypatch, captured_extra_env_vars)
    monkeypatch.setattr(
        patches,
        "_patch_vllm_nemotron_h_fp32_lm_head",
        lambda _logger: fp32_patch_calls.append(True) or True,
    )
    monkeypatch.setattr(
        patches,
        "_patch_vllm_moe_routed_experts_capture",
        lambda _logger, *, required: capture_requirements.append(required) or True,
    )
    fallback_requirements = []
    monkeypatch.setattr(
        patches,
        "_patch_vllm_routed_experts_capture_router_fallback",
        lambda _logger, *, required: fallback_requirements.append(required) or True,
    )

    patches._apply_vllm_patches(
        "py",
        extra_env_vars=["USER_VAR"],
        nemotron_h_fp32_lm_head=enabled,
        require_moe_routed_experts_capture=require_capture,
    )

    assert bool(fp32_patch_calls) is enabled
    assert capture_requirements == [require_capture]
    # The router fallback is required exactly when the capture patch is.
    assert fallback_requirements == [require_capture]
    if enabled:
        assert os.environ[patches.VLLM_NEMOTRON_H_FP32_LM_HEAD_ENV_VAR] == "1"
        assert captured_extra_env_vars == [
            ["USER_VAR", patches.VLLM_NEMOTRON_H_FP32_LM_HEAD_ENV_VAR]
        ]
    else:
        assert patches.VLLM_NEMOTRON_H_FP32_LM_HEAD_ENV_VAR not in os.environ
        assert captured_extra_env_vars == [["USER_VAR"]]


def test_apply_vllm_patches_ignores_ambient_fp32_lm_head_env_toggle(monkeypatch):
    _install_fake_vllm_modules(monkeypatch)
    monkeypatch.setenv(patches.VLLM_NEMOTRON_H_FP32_LM_HEAD_ENV_VAR, "1")
    captured_extra_env_vars = []
    fp32_patch_calls = []
    _stub_non_fp32_vllm_patches(monkeypatch, captured_extra_env_vars)
    monkeypatch.setattr(
        patches,
        "_patch_vllm_nemotron_h_fp32_lm_head",
        lambda _logger: fp32_patch_calls.append(True) or True,
    )

    patches._apply_vllm_patches("py")

    assert fp32_patch_calls == []
    assert patches.VLLM_NEMOTRON_H_FP32_LM_HEAD_ENV_VAR not in os.environ
    assert captured_extra_env_vars == [None]


def test_apply_vllm_patches_raises_when_nemotron_h_fp32_lm_head_patch_fails(
    monkeypatch,
):
    _install_fake_vllm_modules(monkeypatch)
    monkeypatch.delenv(patches.VLLM_NEMOTRON_H_FP32_LM_HEAD_ENV_VAR, raising=False)
    _stub_non_fp32_vllm_patches(monkeypatch, [])
    monkeypatch.setattr(
        patches, "_patch_vllm_nemotron_h_fp32_lm_head", lambda _logger: False
    )

    with pytest.raises(RuntimeError, match="could not be applied"):
        patches._apply_vllm_patches("py", nemotron_h_fp32_lm_head=True)


@pytest.mark.parametrize("require_capture", [False, True])
@pytest.mark.parametrize(
    ("vllm_cfg_overrides", "expected_nemotron_h_fp32_lm_head"),
    [
        ({"env_vars": {"USER_VAR": "value"}, "fp32_lm_head": True}, True),
        (
            {
                "env_vars": {
                    "USER_VAR": "value",
                    VLLM_NEMOTRON_H_FP32_LM_HEAD_ENV_VAR: "1",
                }
            },
            False,
        ),
    ],
)
def test_vllm_worker_threads_nemotron_h_fp32_lm_head_cfg_into_source_patches(
    monkeypatch,
    vllm_cfg_overrides,
    expected_nemotron_h_fp32_lm_head,
    require_capture: bool,
):
    from nemo_rl.models.generation.vllm import vllm_worker

    patch_calls = []
    monkeypatch.setattr(
        vllm_worker,
        "_apply_vllm_patches",
        lambda py,
        *,
        extra_env_vars,
        nemotron_h_fp32_lm_head,
        require_moe_routed_experts_capture: patch_calls.append(
            {
                "py": py,
                "extra_env_vars": extra_env_vars,
                "nemotron_h_fp32_lm_head": nemotron_h_fp32_lm_head,
                "require_moe_routed_experts_capture": require_moe_routed_experts_capture,
            }
        ),
    )

    vllm_worker.BaseVllmGenerationWorker(
        {
            "model_name": "model",
            "vllm_kwargs": {"enable_return_routed_experts": require_capture},
            "vllm_cfg": {
                "tensor_parallel_size": 1,
                "pipeline_parallel_size": 1,
                "expert_parallel_size": 1,
                "gpu_memory_utilization": 0.6,
                "precision": "bfloat16",
                **vllm_cfg_overrides,
            },
        },
        extra_env_vars=["EXPLICIT_VAR"],
    )

    assert patch_calls == [
        {
            "py": sys.executable,
            "extra_env_vars": ["EXPLICIT_VAR"],
            "nemotron_h_fp32_lm_head": expected_nemotron_h_fp32_lm_head,
            "require_moe_routed_experts_capture": require_capture,
        }
    ]


@pytest.mark.parametrize(
    "existing,extra,expected",
    [
        (None, None, "RAY_ENABLE_UV_RUN_RUNTIME_ENV"),
        ("", ["MY_VAR"], "MY_VAR,RAY_ENABLE_UV_RUN_RUNTIME_ENV"),
        # A value the caller already set must survive, not be clobbered.
        ("PRESET", ["MY_VAR"], "MY_VAR,PRESET,RAY_ENABLE_UV_RUN_RUNTIME_ENV"),
        # Duplicates collapse and surrounding whitespace is stripped.
        (
            " PRESET , MY_VAR ",
            ["MY_VAR"],
            "MY_VAR,PRESET,RAY_ENABLE_UV_RUN_RUNTIME_ENV",
        ),
    ],
)
def test_ray_extra_env_vars_merge_is_additive(
    monkeypatch, tmp_path, existing, extra, expected
):
    """vLLM 0.25 replaced the ADDITIONAL_ENV_VARS source patch with this hook.

    It must add to whatever the caller already set rather than overwrite it --
    otherwise user ``extra_env_vars`` silently stop reaching the Ray workers.
    """
    ray_executor = tmp_path / "ray_executor.py"
    ray_executor.write_text("self._init_workers_ray(placement_group)\n")
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _r: str(ray_executor))

    if existing is None:
        monkeypatch.delenv("VLLM_RAY_EXTRA_ENV_VARS_TO_COPY", raising=False)
    else:
        monkeypatch.setenv("VLLM_RAY_EXTRA_ENV_VARS_TO_COPY", existing)

    patches._patch_vllm_init_workers_ray("py", extra)

    assert os.environ["VLLM_RAY_EXTRA_ENV_VARS_TO_COPY"] == expected


def test_init_workers_ray_reports_a_missing_anchor(monkeypatch, tmp_path):
    """A reshaped call site must not be reported as a successful patch."""
    ray_executor = tmp_path / "ray_executor.py"
    ray_executor.write_text("self._init_workers_ray_renamed(placement_group)\n")
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _r: str(ray_executor))
    monkeypatch.delenv("VLLM_RAY_EXTRA_ENV_VARS_TO_COPY", raising=False)

    assert patches._patch_vllm_init_workers_ray("py", None) is False
    # The env merge still has to happen; it is independent of the file patch.
    assert os.environ["VLLM_RAY_EXTRA_ENV_VARS_TO_COPY"] == (
        "RAY_ENABLE_UV_RUN_RUNTIME_ENV"
    )


def test_init_workers_ray_reports_success_and_is_idempotent(monkeypatch, tmp_path):
    """Patching twice against the same file still reports success."""
    ray_executor = tmp_path / "ray_executor.py"
    ray_executor.write_text("self._init_workers_ray(placement_group)\n")
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _r: str(ray_executor))

    assert patches._patch_vllm_init_workers_ray("py-exec", None) is True
    once = ray_executor.read_text()
    assert 'runtime_env={"py_executable": "py-exec"}' in once

    assert patches._patch_vllm_init_workers_ray("py-exec", None) is True
    assert ray_executor.read_text() == once


@pytest.fixture
def patched_scheduler_source(tmp_path, monkeypatch):
    """The installed scheduler.py, unpatched then patched in tmp."""
    copied = write_unpatched_copy(_PAUSE_SOURCE, _PAUSE_PATCH_FN, tmp_path / "sched.py")
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _relative: str(copied))
    patches._patch_vllm_pause_with_held_kv(logging.getLogger(__name__))
    return copied


@pytest.mark.vllm
def test_held_kv_pause_patch_anchor_still_matches_installed_vllm(
    patched_scheduler_source,
):
    """A source edit becomes a silent no-op if upstream rewrites the method."""
    content = patched_scheduler_source.read_text()
    assert _PAUSE_MARKER in content, (
        "the held-KV pause patch did not apply to the installed vLLM; "
        "Scheduler.has_finished_requests() has probably changed upstream. A "
        "prefill/decode in-flight pause then waits for a KV lease again."
    )
    ast.parse(content)  # the edit must leave valid Python


@pytest.mark.vllm
def test_held_kv_pause_patch_is_idempotent(patched_scheduler_source, monkeypatch):
    """Every worker on a node runs the patch against the same file."""
    before = patched_scheduler_source.read_text()
    monkeypatch.setattr(
        patches, "_get_vllm_file", lambda _relative: str(patched_scheduler_source)
    )
    patches._patch_vllm_pause_with_held_kv(logging.getLogger(__name__))
    assert patched_scheduler_source.read_text() == before


def _scheduler_with_has_finished_requests(tmp_path, monkeypatch, *, patch: bool):
    """A stub Scheduler whose has_finished_requests() is the real vLLM method text."""
    old_snippet, _ = patch_snippets(_PAUSE_PATCH_FN)
    source = (
        "import enum\n"
        "\n"
        "\n"
        "class PauseState(enum.IntEnum):\n"
        "    UNPAUSED = 0\n"
        "    PAUSED_NEW = 1\n"
        "    PAUSED_ALL = 2\n"
        "\n"
        "\n"
        "class Scheduler:\n"
        "    def __init__(self, state, finished_req_ids, connector, requests, queued):\n"
        "        self._pause_state = state\n"
        "        self.finished_req_ids = finished_req_ids\n"
        "        self.connector = connector\n"
        "        self.requests = requests\n"
        "        self.waiting = list(range(queued))\n"
        "        self.skipped_waiting = []\n"
        "        self.running = []\n"
        "\n"
        f"{old_snippet}"
        "        num_in_queues = (\n"
        "            len(self.waiting) + len(self.skipped_waiting) + len(self.running)\n"
        "        )\n"
        "        return len(self.requests) > num_in_queues\n"
    )
    file = tmp_path / "scheduler_stub.py"
    file.write_text(source)
    if patch:
        monkeypatch.setattr(patches, "_get_vllm_file", lambda _relative: str(file))
        patches._patch_vllm_pause_with_held_kv(logging.getLogger(__name__))
        assert _PAUSE_MARKER in file.read_text()
    namespace: dict = {}
    exec(compile(file.read_text(), str(file), "exec"), namespace)
    return namespace["Scheduler"], namespace["PauseState"]


@pytest.mark.parametrize(
    ("state", "finished_req_ids", "connector", "held", "expected"),
    [
        # Requests a connector holds after they finish count as work, unless the
        # scheduler is in PAUSED_ALL.
        ("UNPAUSED", set(), object(), True, True),
        ("PAUSED_NEW", set(), object(), True, True),
        ("PAUSED_ALL", set(), object(), True, False),
        # A request that just finished still needs one step to reach the worker.
        ("PAUSED_ALL", {"req"}, object(), True, True),
        # Without a connector nothing is held, whatever the state.
        ("PAUSED_ALL", set(), None, True, False),
        ("UNPAUSED", set(), object(), False, False),
    ],
)
def test_held_kv_pause_patch_changes_only_the_paused_all_state(
    tmp_path, monkeypatch, state, finished_req_ids, connector, held, expected
):
    scheduler_cls, pause_state = _scheduler_with_has_finished_requests(
        tmp_path, monkeypatch, patch=True
    )
    # "held" means a finished request is still in self.requests but in no queue.
    scheduler = scheduler_cls(
        pause_state[state],
        finished_req_ids,
        connector,
        requests={"held-request"} if held else set(),
        queued=0,
    )

    assert scheduler.has_finished_requests() is expected


def test_held_kv_pause_patch_is_needed_without_it(tmp_path, monkeypatch):
    """Negative control: unpatched vLLM waits on held KV even when PAUSED_ALL."""
    scheduler_cls, pause_state = _scheduler_with_has_finished_requests(
        tmp_path, monkeypatch, patch=False
    )
    scheduler = scheduler_cls(
        pause_state.PAUSED_ALL, set(), object(), requests={"held-request"}, queued=0
    )

    assert scheduler.has_finished_requests() is True


def test_held_kv_pause_patch_warns_on_unknown_source(monkeypatch, tmp_path, caplog):
    scheduler_source = tmp_path / "scheduler.py"
    scheduler_source.write_text("class Scheduler:\n    pass\n")
    monkeypatch.setattr(
        patches, "_get_vllm_file", lambda _relative: str(scheduler_source)
    )

    with caplog.at_level(logging.WARNING):
        patches._patch_vllm_pause_with_held_kv(logging.getLogger(__name__))

    assert scheduler_source.read_text() == "class Scheduler:\n    pass\n"
    assert "has_finished_requests() not found" in caplog.text


def _unpatched_reset_source(tmp_path):
    """The installed scheduler.py with the held-KV reset patch reversed."""
    from pathlib import Path

    old_raise, new_raise = patch_snippets(_RESET_PATCH_FN, "old_raise", "new_raise")
    old_anchor, new_method = patch_snippets(
        _RESET_PATCH_FN, "old_method_anchor", "new_method"
    )
    content = Path(patches._get_vllm_file(_PAUSE_SOURCE)).read_text()
    content = content.replace(new_method + old_anchor, old_anchor, 1)
    content = content.replace(new_raise, old_raise, 1)
    assert old_raise in content and old_anchor in content, (
        "the installed scheduler.py has neither form of the held-KV reset anchor"
    )
    assert _RESET_MARKER not in content
    destination = tmp_path / "sched_reset.py"
    destination.write_text(content)
    return destination


@pytest.fixture
def patched_reset_source(tmp_path, monkeypatch):
    copied = _unpatched_reset_source(tmp_path)
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _relative: str(copied))
    assert patches._patch_vllm_reset_cache_with_held_kv(logging.getLogger(__name__))
    return copied


@pytest.mark.vllm
def test_held_kv_reset_patch_anchor_still_matches_installed_vllm(patched_reset_source):
    """A source edit becomes a silent no-op if upstream rewrites the reset."""
    content = patched_reset_source.read_text()
    assert _RESET_MARKER in content, (
        "the held-KV reset patch did not apply to the installed vLLM; "
        "Scheduler.reset_prefix_cache() has probably changed upstream. A cache "
        "clear on a prefill/decode fleet then raises an error."
    )
    ast.parse(content)  # the edit must leave valid Python


@pytest.mark.vllm
def test_held_kv_reset_patch_is_idempotent(patched_reset_source, monkeypatch):
    before = patched_reset_source.read_text()
    monkeypatch.setattr(
        patches, "_get_vllm_file", lambda _relative: str(patched_reset_source)
    )
    assert patches._patch_vllm_reset_cache_with_held_kv(logging.getLogger(__name__))
    assert patched_reset_source.read_text() == before


def _scheduler_with_reset_prefix_cache(tmp_path, monkeypatch, *, patch: bool):
    """A stub Scheduler around the real raise branch of reset_prefix_cache()."""
    old_raise, _ = patch_snippets(_RESET_PATCH_FN, "old_raise", "new_raise")
    source = (
        "import enum\n"
        "import logging\n"
        "\n"
        "logger = logging.getLogger('stub')\n"
        "\n"
        "\n"
        "class PauseState(enum.IntEnum):\n"
        "    UNPAUSED = 0\n"
        "    PAUSED_ALL = 2\n"
        "\n"
        "\n"
        "class Scheduler:\n"
        "    def reset_prefix_cache(\n"
        "        self, reset_running_requests=False, reset_connector=False\n"
        "    ):\n"
        "        if reset_running_requests:\n"
        "            while self.running:\n"
        "                self.preempted.append(self.running.pop())\n"
        f"{old_raise}"
        "        return reset_successful\n"
        "\n"
        "    def reset_connector_cache(self) -> bool:\n"
        "        return True\n"
    )
    file = tmp_path / "reset_stub.py"
    file.write_text(source)
    if patch:
        monkeypatch.setattr(patches, "_get_vllm_file", lambda _relative: str(file))
        assert patches._patch_vllm_reset_cache_with_held_kv(logging.getLogger(__name__))
    namespace: dict = {}
    exec(compile(file.read_text(), str(file), "exec"), namespace)
    return namespace["Scheduler"]


class _StubBlock:
    def __init__(self):
        self.cached = True

    def reset_hash(self):
        self.cached = False


def _stub_scheduler(
    scheduler_cls, *, reset_succeeds, connector, running=(), paused_state=0
):
    from types import SimpleNamespace

    blocks = [_StubBlock() for _ in range(3)]
    pool = SimpleNamespace(
        cached_block_hash_to_block={"hash": 1},
        cached_block_hashes_by_block={1: "hash"},
        blocks=blocks,
        metrics_collector=None,
        enable_kv_cache_events=False,
        kv_event_queue=[],
    )
    scheduler = scheduler_cls()
    scheduler.kv_cache_manager = SimpleNamespace(
        block_pool=pool,
        log_stats=False,
        prefix_cache_stats=None,
        reset_prefix_cache=lambda: reset_succeeds,
    )
    scheduler.connector = connector
    scheduler.running = list(running)
    scheduler.preempted = []
    scheduler._pause_state = paused_state
    return scheduler, pool, blocks


def test_held_kv_reset_patch_forces_the_reset_when_a_connector_holds_blocks(
    tmp_path, monkeypatch
):
    scheduler_cls = _scheduler_with_reset_prefix_cache(
        tmp_path, monkeypatch, patch=True
    )
    scheduler, pool, blocks = _stub_scheduler(
        scheduler_cls, reset_succeeds=False, connector=object(), running=["a", "b"]
    )

    assert scheduler.reset_prefix_cache(reset_running_requests=True) is True

    # Every cached hash is gone, so no new request hits the old weights' KV, and
    # the running requests were still preempted first.
    assert pool.cached_block_hash_to_block == {}
    assert pool.cached_block_hashes_by_block == {}
    assert not any(block.cached for block in blocks)
    assert scheduler.preempted == ["b", "a"]


@pytest.mark.parametrize(("paused_state", "armed"), [(0, False), (2, True)])
def test_held_kv_reset_patch_arms_the_stale_remote_kv_patch_only_in_a_pause(
    tmp_path, monkeypatch, paused_state, armed
):
    scheduler_cls = _scheduler_with_reset_prefix_cache(
        tmp_path, monkeypatch, patch=True
    )
    scheduler, _, _ = _stub_scheduler(
        scheduler_cls,
        reset_succeeds=False,
        connector=object(),
        paused_state=paused_state,
    )

    scheduler.reset_prefix_cache(reset_running_requests=True)

    assert getattr(scheduler, "_nemo_rl_reset_in_pause", False) is armed


def test_held_kv_reset_patch_still_raises_without_a_connector(tmp_path, monkeypatch):
    scheduler_cls = _scheduler_with_reset_prefix_cache(
        tmp_path, monkeypatch, patch=True
    )
    scheduler, _, blocks = _stub_scheduler(
        scheduler_cls, reset_succeeds=False, connector=None
    )

    # Without a connector nothing explains the held blocks, so vLLM must not hide it.
    with pytest.raises(RuntimeError, match="Failed to reset KV cache"):
        scheduler.reset_prefix_cache(reset_running_requests=True)
    assert all(block.cached for block in blocks)


def test_held_kv_reset_patch_leaves_a_successful_reset_alone(tmp_path, monkeypatch):
    scheduler_cls = _scheduler_with_reset_prefix_cache(
        tmp_path, monkeypatch, patch=True
    )
    scheduler, _, blocks = _stub_scheduler(
        scheduler_cls, reset_succeeds=True, connector=object()
    )

    assert scheduler.reset_prefix_cache(reset_running_requests=True) is True
    # vLLM's own reset cleared the cache; the patch adds nothing.
    assert all(block.cached for block in blocks)


def test_held_kv_reset_is_needed_without_the_patch(tmp_path, monkeypatch):
    """Negative control: unpatched vLLM raises when a connector holds blocks."""
    scheduler_cls = _scheduler_with_reset_prefix_cache(
        tmp_path, monkeypatch, patch=False
    )
    scheduler, _, _ = _stub_scheduler(
        scheduler_cls, reset_succeeds=False, connector=object()
    )

    with pytest.raises(RuntimeError, match="Failed to reset KV cache"):
        scheduler.reset_prefix_cache(reset_running_requests=True)


def test_held_kv_reset_patch_warns_on_unknown_source(monkeypatch, tmp_path, caplog):
    scheduler_source = tmp_path / "scheduler.py"
    scheduler_source.write_text("class Scheduler:\n    pass\n")
    monkeypatch.setattr(
        patches, "_get_vllm_file", lambda _relative: str(scheduler_source)
    )

    with caplog.at_level(logging.WARNING):
        applied = patches._patch_vllm_reset_cache_with_held_kv(
            logging.getLogger(__name__)
        )

    assert applied is False
    assert scheduler_source.read_text() == "class Scheduler:\n    pass\n"
    assert "reset_prefix_cache() not found" in caplog.text


def test_require_held_kv_patches_raises_when_an_anchor_is_missing(
    monkeypatch, tmp_path
):
    scheduler_source = tmp_path / "scheduler.py"
    scheduler_source.write_text("class Scheduler:\n    pass\n")
    monkeypatch.setattr(
        patches, "_get_vllm_file", lambda _relative: str(scheduler_source)
    )
    vllm_logger = types.ModuleType("vllm.logger")
    vllm_logger.init_logger = logging.getLogger
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.logger", vllm_logger)

    with pytest.raises(
        RuntimeError, match="pause_with_held_kv.*reset_cache_with_held_kv"
    ):
        patches.require_held_kv_patches()


def test_the_first_held_kv_patch_reports_whether_it_applied(monkeypatch, tmp_path):
    scheduler_source = tmp_path / "scheduler.py"
    scheduler_source.write_text("class Scheduler:\n    pass\n")
    monkeypatch.setattr(
        patches, "_get_vllm_file", lambda _relative: str(scheduler_source)
    )

    assert patches._patch_vllm_pause_with_held_kv(logging.getLogger(__name__)) is False


def _unpatched_stale_source(tmp_path):
    """The installed scheduler.py with the stale remote KV patch reversed."""
    from pathlib import Path

    old_pause, new_pause = patch_snippets(_STALE_PATCH_FN, "old_pause", "new_pause")
    old_remote, new_remote = patch_snippets(_STALE_PATCH_FN, "old_remote", "new_remote")
    content = Path(patches._get_vllm_file(_PAUSE_SOURCE)).read_text()
    content = content.replace(new_remote, old_remote, 1)
    content = content.replace(new_pause, old_pause, 1)
    assert old_pause in content and old_remote in content, (
        "the installed scheduler.py has neither form of the stale remote KV anchors"
    )
    assert _STALE_MARKER not in content
    destination = tmp_path / "sched_stale.py"
    destination.write_text(content)
    return destination


@pytest.fixture
def patched_stale_source(tmp_path, monkeypatch):
    copied = _unpatched_stale_source(tmp_path)
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _relative: str(copied))
    assert patches._patch_vllm_stale_remote_kv_after_reset(logging.getLogger(__name__))
    return copied


@pytest.mark.vllm
def test_stale_remote_kv_patch_anchor_still_matches_installed_vllm(
    patched_stale_source,
):
    content = patched_stale_source.read_text()
    assert _STALE_MARKER in content, (
        "the stale remote KV patch did not apply to the installed vLLM; "
        "set_pause_state() or _update_waiting_for_remote_kv() has probably "
        "changed upstream. A decode engine can then serve KV from the old weights "
        "after a cache clear."
    )
    ast.parse(content)


@pytest.mark.vllm
def test_stale_remote_kv_patch_is_idempotent(patched_stale_source, monkeypatch):
    before = patched_stale_source.read_text()
    monkeypatch.setattr(
        patches, "_get_vllm_file", lambda _relative: str(patched_stale_source)
    )
    assert patches._patch_vllm_stale_remote_kv_after_reset(logging.getLogger(__name__))
    assert patched_stale_source.read_text() == before


def _stale_stub_scheduler(tmp_path, monkeypatch):
    """A stub Scheduler around the real anchors of the stale remote KV patch."""
    old_pause, _ = patch_snippets(_STALE_PATCH_FN, "old_pause", "new_pause")
    old_remote, _ = patch_snippets(_STALE_PATCH_FN, "old_remote", "new_remote")
    source = (
        "import enum\n"
        "import logging\n"
        "import time\n"
        "\n"
        "logger = logging.getLogger('stub')\n"
        "\n"
        "\n"
        "class PauseState(enum.IntEnum):\n"
        "    UNPAUSED = 0\n"
        "    PAUSED_ALL = 2\n"
        "\n"
        "\n"
        "class Request:\n"
        "    pass\n"
        "\n"
        "\n"
        "class Scheduler:\n"
        "    def set_pause_state(self, pause_state):\n"
        f"{old_pause}"
        "\n"
        f"{old_remote}"
        "        self.finished.append(request.request_id)\n"
    )
    file = tmp_path / "stale_stub.py"
    file.write_text(source)
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _relative: str(file))
    assert patches._patch_vllm_stale_remote_kv_after_reset(logging.getLogger(__name__))
    namespace: dict = {}
    exec(compile(file.read_text(), str(file), "exec"), namespace)
    return namespace["Scheduler"], namespace["PauseState"]


class _EvictingPool:
    def __init__(self):
        self.evicted = []

    def _maybe_evict_cached_block(self, block):
        self.evicted.append(block)
        return True


def _stale_scheduler(scheduler_cls):
    scheduler = scheduler_cls()
    scheduler.finished = []
    scheduler._pause_state = 2
    pool = _EvictingPool()
    scheduler.kv_cache_manager = SimpleNamespace(
        block_pool=pool,
        get_blocks=lambda _request_id: SimpleNamespace(blocks=(["b0", "b1"], ["b2"])),
    )
    return scheduler, pool


def _request(arrival_time, num_computed_tokens=16):
    return SimpleNamespace(
        request_id="req",
        arrival_time=arrival_time,
        num_computed_tokens=num_computed_tokens,
    )


def test_stale_remote_kv_patch_uncaches_loads_of_requests_that_arrived_before_resume(
    tmp_path, monkeypatch
):
    scheduler_cls, pause_state = _stale_stub_scheduler(tmp_path, monkeypatch)
    scheduler, pool = _stale_scheduler(scheduler_cls)
    scheduler._nemo_rl_reset_in_pause = True  # armed by a forced reset

    scheduler.set_pause_state(pause_state.UNPAUSED)
    resumed_at = scheduler._nemo_rl_stale_remote_kv_before

    assert scheduler._nemo_rl_reset_in_pause is False
    scheduler._update_waiting_for_remote_kv(_request(resumed_at - 1.0))
    assert pool.evicted == ["b0", "b1", "b2"]
    # The wrapped vLLM method still runs.
    assert scheduler.finished == ["req"]

    # A request that arrived after the resume has KV that the new weights computed.
    pool.evicted.clear()
    scheduler._update_waiting_for_remote_kv(_request(resumed_at + 1.0))
    assert pool.evicted == []


def test_stale_remote_kv_patch_does_nothing_without_a_reset_in_the_pause(
    tmp_path, monkeypatch
):
    scheduler_cls, pause_state = _stale_stub_scheduler(tmp_path, monkeypatch)
    scheduler, pool = _stale_scheduler(scheduler_cls)

    scheduler.set_pause_state(pause_state.UNPAUSED)
    scheduler._update_waiting_for_remote_kv(_request(1.0))

    assert pool.evicted == []
    assert scheduler._pause_state == pause_state.UNPAUSED


def test_stale_remote_kv_patch_skips_a_load_that_left_no_blocks(tmp_path, monkeypatch):
    scheduler_cls, _ = _stale_stub_scheduler(tmp_path, monkeypatch)
    scheduler, pool = _stale_scheduler(scheduler_cls)
    scheduler._nemo_rl_stale_remote_kv_before = 1e18

    # A failed load with no valid tokens frees the request's blocks.
    scheduler._update_waiting_for_remote_kv(_request(0.0, num_computed_tokens=0))

    assert pool.evicted == []


def test_stale_remote_kv_patch_warns_on_unknown_source(monkeypatch, tmp_path, caplog):
    scheduler_source = tmp_path / "scheduler.py"
    scheduler_source.write_text("class Scheduler:\n    pass\n")
    monkeypatch.setattr(
        patches, "_get_vllm_file", lambda _relative: str(scheduler_source)
    )

    with caplog.at_level(logging.WARNING):
        applied = patches._patch_vllm_stale_remote_kv_after_reset(
            logging.getLogger(__name__)
        )

    assert applied is False
    assert scheduler_source.read_text() == "class Scheduler:\n    pass\n"
    assert "_update_waiting_for_remote_kv() not found" in caplog.text
