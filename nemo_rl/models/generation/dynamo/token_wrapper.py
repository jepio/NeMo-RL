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

"""OpenAI-compatible token wrapper for NeMo-Gym traffic to Dynamo."""

import asyncio
import json
import threading
import time
from contextlib import asynccontextmanager
from copy import deepcopy
from typing import Any, Optional

from nemo_rl.distributed.virtual_cluster import (
    DEFAULT_DYNAMO_HTTP_PORT_RANGE_HIGH,
    DEFAULT_DYNAMO_HTTP_PORT_RANGE_LOW,
    _get_free_port_local,
    _get_node_ip_local,
)
from nemo_rl.models.generation.openai_server_utils import replace_prefix_tokens

try:
    import orjson
except ImportError:  # pragma: no cover - orjson is a NeMo RL dependency
    orjson = None


# The HTTP header that Dynamo's frontend turns into the session of a request. The
# router binds a session to one worker when session affinity is on
# (``--router-session-affinity-ttl-secs``), so every turn of a rollout reaches the
# engine that holds its prefix.
# Epoch seconds are about 1.8e9, which fits a signed 32-bit integer, but the
# request priority is a negated age in seconds from this base, so it stays far
# from the limit.
_AGE_PRIORITY_EPOCH_S = 1_700_000_000
_MAX_TRACKED_SESSIONS = 100_000


def age_priority(first_seen_epoch_s: float) -> int:
    """Return the Dynamo agent-hint priority of a conversation first seen at a time.

    Dynamo gives a request with a higher agent-hint priority an earlier place, and
    its vLLM handler hands vLLM the negated value, where lower runs first. The
    priority falls as the first request of a conversation gets newer, so older
    conversations run first.
    """
    return -int(first_seen_epoch_s - _AGE_PRIORITY_EPOCH_S)


DYNAMO_SESSION_ID_HEADER = "x-dynamo-session-id"


def pop_gym_conversation_id(request_body: dict[str, Any]) -> Optional[str]:
    """Remove NeMo-Gym's ``conversation_params`` from the body and return its id.

    Gym adds ``conversation_params.conversation_id``, a TensorRT-LLM extension that
    is not part of the OpenAI schema, when ``forward_session_id_as_conversation_id``
    is on. Dynamo takes the session as a header instead, so the field must not go on.
    """
    params = request_body.pop("conversation_params", None)
    if not isinstance(params, dict):
        return None
    conversation_id = params.get("conversation_id")
    if isinstance(conversation_id, (str, int)) and str(conversation_id).strip():
        return str(conversation_id).strip()
    return None


def _loads(raw: bytes | str) -> Any:
    """Parse JSON. orjson is several times faster than json on long token lists."""
    return orjson.loads(raw) if orjson is not None else json.loads(raw)


def _dumps(value: Any) -> bytes:
    """Serialize JSON to bytes, with orjson when it is installed."""
    if orjson is not None:
        return orjson.dumps(value)
    return json.dumps(value).encode()


_GYM_TOKEN_METADATA_FIELDS = (
    "prompt_token_ids",
    "generation_token_ids",
    "generation_log_probs",
)
_TOOL_ARGUMENT_MAPPING_ERROR = "Can only get item pairs from a mapping."


def _coerce_token_id_list(value: Any, field_name: str) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list of token IDs.")
    # A prompt of tens of thousands of IDs arrives on every request, and the common
    # case needs no conversion. The check runs in C, while the loop below holds the GIL
    # for every element.
    if set(map(type, value)) <= {int}:
        return value
    try:
        return [int(token_id) for token_id in value]
    except (TypeError, ValueError) as e:
        raise ValueError(f"{field_name} must contain only integer token IDs.") from e


def _coerce_logprob_list(value: Any, field_name: str) -> list[float]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list of numeric log probabilities.")
    logprobs: list[float] = []
    for index, logprob in enumerate(value):
        if not isinstance(logprob, (int, float)) or isinstance(logprob, bool):
            raise ValueError(f"{field_name}[{index}] must be numeric.")
        logprobs.append(float(logprob))
    return logprobs


def _strip_gym_token_metadata(messages: list[Any]) -> list[Any]:
    """Return the messages without the token metadata that Gym attached.

    Only a message that carries metadata is copied, and the copy is shallow. A deep
    copy of a long conversation, with the token lists of its latest turn, held the
    GIL for a third of the wrapper's time.
    """
    stripped_messages = []
    for message in messages:
        if isinstance(message, dict) and any(
            field in message for field in _GYM_TOKEN_METADATA_FIELDS
        ):
            message = {
                key: value
                for key, value in message.items()
                if key not in _GYM_TOKEN_METADATA_FIELDS
            }
        stripped_messages.append(message)
    return stripped_messages


def _chat_template_kwargs(
    request_body: dict[str, Any],
    tokenizer_chat_template_kwargs: Optional[dict[str, Any]],
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if tokenizer_chat_template_kwargs is not None:
        if not isinstance(tokenizer_chat_template_kwargs, dict):
            raise ValueError("tokenizer chat_template_kwargs must be a JSON object.")
        kwargs.update(tokenizer_chat_template_kwargs)

    request_kwargs = request_body.get("chat_template_kwargs")
    if request_kwargs is not None:
        if not isinstance(request_kwargs, dict):
            raise ValueError("chat_template_kwargs must be a JSON object.")
        kwargs.update(request_kwargs)

    if "reasoning_effort" in request_body:
        kwargs["reasoning_effort"] = request_body["reasoning_effort"]
    return kwargs


def _request_add_generation_prompt(request_body: dict[str, Any]) -> bool:
    if "add_generation_prompt" in request_body:
        return bool(request_body["add_generation_prompt"])
    return not bool(request_body.get("continue_final_message", False))


def _apply_chat_template(
    *,
    tokenizer: Any,
    request_body: dict[str, Any],
    messages: list[Any],
    tokenizer_chat_template_kwargs: Optional[dict[str, Any]],
    exclude_tools_when_tool_choice_none: bool,
    add_generation_prompt: bool,
    tokenize: bool,
) -> Any:
    tools = request_body.get("tools")
    if (
        exclude_tools_when_tool_choice_none
        and request_body.get("tool_choice") == "none"
    ):
        tools = None

    apply_chat_template = type(tokenizer).apply_chat_template
    return apply_chat_template(
        tokenizer,
        messages,
        tools=tools,
        documents=request_body.get("documents"),
        chat_template=request_body.get("chat_template"),
        add_generation_prompt=add_generation_prompt,
        continue_final_message=bool(request_body.get("continue_final_message", False)),
        tokenize=tokenize,
        return_tensors=None,
        return_dict=False,
        **_chat_template_kwargs(request_body, tokenizer_chat_template_kwargs),
    )


def _render_prompt_token_ids(
    *,
    tokenizer: Any,
    request_body: dict[str, Any],
    messages: list[Any],
    tokenizer_chat_template_kwargs: Optional[dict[str, Any]],
    exclude_tools_when_tool_choice_none: bool,
    add_generation_prompt: bool,
) -> list[int]:
    token_ids = _apply_chat_template(
        tokenizer=tokenizer,
        request_body=request_body,
        messages=messages,
        tokenizer_chat_template_kwargs=tokenizer_chat_template_kwargs,
        exclude_tools_when_tool_choice_none=exclude_tools_when_tool_choice_none,
        add_generation_prompt=add_generation_prompt,
        tokenize=True,
    )

    if isinstance(token_ids, list) and (
        not token_ids or not isinstance(token_ids[0], list)
    ):
        return _coerce_token_id_list(token_ids, "prompt token IDs")
    if isinstance(token_ids, list) and len(token_ids) == 1:
        return _coerce_token_id_list(token_ids[0], "prompt token IDs")
    raise ValueError(
        "Dynamo token wrapper expected chat template rendering to return one "
        "list of prompt token IDs."
    )


def _render_prompt_token_ids_with_optional_prefix(
    *,
    tokenizer: Any,
    request_body: dict[str, Any],
    messages: list[Any],
    tokenizer_chat_template_kwargs: Optional[dict[str, Any]],
    exclude_tools_when_tool_choice_none: bool,
    add_generation_prompt: bool,
    assistant_index: int | None,
) -> tuple[list[int], list[int] | None]:
    full_prompt_token_ids = _render_prompt_token_ids(
        tokenizer=tokenizer,
        request_body=request_body,
        messages=messages,
        tokenizer_chat_template_kwargs=tokenizer_chat_template_kwargs,
        exclude_tools_when_tool_choice_none=exclude_tools_when_tool_choice_none,
        add_generation_prompt=add_generation_prompt,
    )
    if assistant_index is None:
        return full_prompt_token_ids, None

    template_prefix_token_ids = _render_prompt_token_ids(
        tokenizer=tokenizer,
        request_body=request_body,
        messages=messages[: assistant_index + 1],
        tokenizer_chat_template_kwargs=tokenizer_chat_template_kwargs,
        exclude_tools_when_tool_choice_none=exclude_tools_when_tool_choice_none,
        add_generation_prompt=False,
    )
    return full_prompt_token_ids, template_prefix_token_ids


def _latest_tokenized_assistant_index(messages: list[Any]) -> Optional[int]:
    for index in reversed(range(len(messages))):
        message = messages[index]
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        if (
            message.get("prompt_token_ids") is not None
            and message.get("generation_token_ids") is not None
        ):
            return index
    return None


def _derive_required_prefix_token_ids(messages: list[Any]) -> list[int] | None:
    """Return the exact prompt and generation IDs from the latest model turn."""
    assistant_index = _latest_tokenized_assistant_index(messages)
    if assistant_index is None:
        return None
    message = messages[assistant_index]
    if not isinstance(message, dict):
        return None
    return _coerce_token_id_list(
        message["prompt_token_ids"], "prompt_token_ids"
    ) + _coerce_token_id_list(message["generation_token_ids"], "generation_token_ids")


def _normalize_tool_arguments_for_template(
    messages: list[Any], *, before_index: int
) -> None:
    """Make OpenAI tool calls renderable by model chat templates.

    OpenAI chat messages carry ``function.arguments`` as a JSON string, while
    some model templates iterate those arguments as a mapping. Normalize only
    the local template copy; the request forwarded to Dynamo retains its
    original OpenAI payload.
    """
    for message in messages[:before_index]:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function", tool_call)
            if not isinstance(function, dict):
                continue
            arguments = function.get("arguments")
            if not isinstance(arguments, str):
                continue
            try:
                parsed_arguments = json.loads(arguments)
            except json.JSONDecodeError:
                parsed_arguments = {}
            function["arguments"] = (
                parsed_arguments if isinstance(parsed_arguments, dict) else {}
            )


def _validate_engine_data(
    response_body: dict[str, Any],
) -> tuple[list[int], list[int], list[float]]:
    nvext = response_body.get("nvext")
    engine_data = nvext.get("engine_data") if isinstance(nvext, dict) else None
    if not isinstance(engine_data, dict):
        raise ValueError("Dynamo response did not include nvext.engine_data.")

    prompt_token_ids = _coerce_token_id_list(
        engine_data.get("prompt_token_ids"),
        "nvext.engine_data.prompt_token_ids",
    )
    completion_token_ids = _coerce_token_id_list(
        engine_data.get("completion_token_ids"),
        "nvext.engine_data.completion_token_ids",
    )
    completion_logprobs = _coerce_logprob_list(
        engine_data.get("completion_logprobs"),
        "nvext.engine_data.completion_logprobs",
    )
    if len(completion_logprobs) != len(completion_token_ids):
        raise ValueError(
            "Dynamo response returned "
            f"{len(completion_logprobs)} generation log probabilities for "
            f"{len(completion_token_ids)} generation token IDs."
        )
    return prompt_token_ids, completion_token_ids, completion_logprobs


def _inject_gym_token_metadata(response_body: dict[str, Any]) -> None:
    """Expose Dynamo engine metadata on the message fields consumed by Gym."""
    (
        prompt_token_ids,
        generation_token_ids,
        generation_logprobs,
    ) = _validate_engine_data(response_body)
    choices = response_body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("Dynamo response did not include choices[0].")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise ValueError("Dynamo response choices[0] must be a JSON object.")
    message = choice.get("message")
    if not isinstance(message, dict):
        raise ValueError("Dynamo response choices[0].message must be a JSON object.")
    message["prompt_token_ids"] = prompt_token_ids
    message["generation_token_ids"] = generation_token_ids
    message["generation_log_probs"] = generation_logprobs
    # The same IDs now sit on the message. Keeping engine_data would carry the whole
    # prompt twice through the response, and through every JSON pass after this one.
    nvext = response_body.get("nvext")
    if isinstance(nvext, dict):
        nvext.pop("engine_data", None)


def prepare_dynamo_chat_completion_request(
    request_body: dict[str, Any],
    *,
    tokenizer: Any,
    tokenizer_chat_template_kwargs: Optional[dict[str, Any]] = None,
    exclude_tools_when_tool_choice_none: bool,
    max_model_len: Optional[int] = None,
) -> dict[str, Any]:
    """Prepare a NeMo-Gym chat-completion request for Dynamo token input.

    Args:
        request_body: The chat-completion request that NeMo-Gym sends.
        tokenizer: Renders the chat template into prompt token ids.
        tokenizer_chat_template_kwargs: Default chat-template arguments.
        exclude_tools_when_tool_choice_none: Drop tools from the prompt when
            ``tool_choice`` is ``none``.
        max_model_len: The engines' context length. Dynamo rejects a request with
            HTTP 400 when the prompt plus the requested output exceeds it, while the
            vLLM server shortens the output to fit. When set, the output limit is
            lowered to the room that is left, as the vLLM server does.

    Raises:
        ValueError: If the request cannot be prepared, or the prompt alone does not
            fit in ``max_model_len``.
    """
    if request_body.get("stream"):
        raise ValueError("Dynamo native token wrapper does not support stream=True.")

    n = request_body.get("n", 1)
    if n is not None and int(n) != 1:
        raise ValueError("Dynamo native token wrapper currently supports only n=1.")

    messages = request_body.get("messages")
    if not isinstance(messages, list):
        raise ValueError("Dynamo token wrapper requires chat-completion messages.")

    # Shallow copy: only top-level keys of the copy change below.
    prepared_body = dict(request_body)
    stripped_messages = _strip_gym_token_metadata(messages)
    prepared_body["messages"] = stripped_messages
    prepared_body.pop("required_prefix_token_ids", None)

    required_prefix_token_ids = _derive_required_prefix_token_ids(messages)
    add_generation_prompt = _request_add_generation_prompt(prepared_body)
    template_messages = stripped_messages
    assistant_index: int | None = None
    if required_prefix_token_ids is not None:
        assistant_index = _latest_tokenized_assistant_index(messages)
        if assistant_index is None:
            raise ValueError(
                "Dynamo prefix token metadata must be attached to an assistant message."
            )

    try:
        (
            full_prompt_token_ids,
            template_prefix_token_ids,
        ) = _render_prompt_token_ids_with_optional_prefix(
            tokenizer=tokenizer,
            request_body=prepared_body,
            messages=template_messages,
            tokenizer_chat_template_kwargs=tokenizer_chat_template_kwargs,
            exclude_tools_when_tool_choice_none=exclude_tools_when_tool_choice_none,
            add_generation_prompt=add_generation_prompt,
            assistant_index=assistant_index,
        )
    except TypeError as e:
        if str(e) != _TOOL_ARGUMENT_MAPPING_ERROR:
            raise
        # The normalization rewrites tool calls in place, and the request that goes to
        # Dynamo must keep its OpenAI payload, so it works on a private copy.
        template_messages = deepcopy(stripped_messages)
        _normalize_tool_arguments_for_template(
            template_messages, before_index=len(template_messages)
        )
        (
            full_prompt_token_ids,
            template_prefix_token_ids,
        ) = _render_prompt_token_ids_with_optional_prefix(
            tokenizer=tokenizer,
            request_body=prepared_body,
            messages=template_messages,
            tokenizer_chat_template_kwargs=tokenizer_chat_template_kwargs,
            exclude_tools_when_tool_choice_none=exclude_tools_when_tool_choice_none,
            add_generation_prompt=add_generation_prompt,
            assistant_index=assistant_index,
        )

    if required_prefix_token_ids is not None:
        assert template_prefix_token_ids is not None
        full_prompt_token_ids = replace_prefix_tokens(
            tokenizer,
            model_prefix_token_ids=required_prefix_token_ids,
            template_prefix_token_ids=template_prefix_token_ids,
            template_token_ids=full_prompt_token_ids,
        )

    nvext = prepared_body.get("nvext")
    if nvext is None:
        nvext = {}
    if not isinstance(nvext, dict):
        raise ValueError("nvext must be a JSON object.")
    nvext = dict(nvext)
    extra_fields = nvext.get("extra_fields", [])
    if not isinstance(extra_fields, list):
        raise ValueError("nvext.extra_fields must be a JSON list.")
    nvext["extra_fields"] = list(dict.fromkeys([*extra_fields, "engine_data"]))
    nvext["token_data"] = full_prompt_token_ids
    prepared_body["nvext"] = nvext

    if max_model_len is not None:
        _clamp_output_tokens_to_context(
            prepared_body,
            prompt_tokens=len(full_prompt_token_ids),
            max_model_len=max_model_len,
        )

    return prepared_body


def _clamp_output_tokens_to_context(
    body: dict[str, Any], *, prompt_tokens: int, max_model_len: int
) -> None:
    """Lower the requested output length to what the context length leaves."""
    remaining = max_model_len - prompt_tokens
    if remaining < 1:
        raise ValueError(
            f"This model's maximum context length is {max_model_len} tokens. "
            f"However, your request has {prompt_tokens} input tokens. "
            "Please reduce the length of the input messages."
        )
    for key in ("max_completion_tokens", "max_tokens"):
        requested = body.get(key)
        if requested is not None and int(requested) > remaining:
            body[key] = remaining


class DynamoTokenWrapperHost:
    """Run one token wrapper server inside a Ray actor, with its own tokenizer.

    The wrapper tokenizes every rollout request in Python. Hosted in the driver it
    shares the driver's GIL, so several of these, each in its own process, are what
    scales the rate of requests. Wrap the class with ``ray.remote`` to use it.
    """

    def __init__(
        self,
        *,
        tokenizer_config: dict[str, Any],
        dynamo_frontend_base_url: str,
        tokenizer_chat_template_kwargs: Optional[dict[str, Any]],
        exclude_tools_when_tool_choice_none: bool,
        request_timeout_s: Optional[float],
        max_model_len: Optional[int],
        age_priority: bool,
        slow_session_log_s: Optional[float],
    ) -> None:
        from nemo_rl.algorithms.utils import get_tokenizer
        from nemo_rl.utils.fastokens import maybe_patch_fastokens

        maybe_patch_fastokens(bool(tokenizer_config.get("use_fastokens")))
        self._server = DynamoTokenWrapperServer(
            dynamo_frontend_base_url=dynamo_frontend_base_url,
            tokenizer=get_tokenizer(tokenizer_config),
            tokenizer_chat_template_kwargs=tokenizer_chat_template_kwargs,
            exclude_tools_when_tool_choice_none=exclude_tools_when_tool_choice_none,
            request_timeout_s=request_timeout_s,
            max_model_len=max_model_len,
            age_priority=age_priority,
            slow_session_log_s=slow_session_log_s,
        )

    def start(self) -> str:
        """Start the server and return its base URL."""
        return self._server.start()

    def shutdown(self) -> None:
        """Stop the server."""
        self._server.shutdown()


class DynamoTokenWrapperServer:
    """Small HTTP server that supplies tokenized chat prompts to Dynamo."""

    def __init__(
        self,
        *,
        dynamo_frontend_base_url: str,
        tokenizer: Any,
        tokenizer_chat_template_kwargs: Optional[dict[str, Any]],
        exclude_tools_when_tool_choice_none: bool,
        request_timeout_s: Optional[float],
        age_priority: bool,
        slow_session_log_s: Optional[float] = None,
        max_model_len: Optional[int] = None,
    ) -> None:
        self.dynamo_frontend_base_url = dynamo_frontend_base_url
        self.age_priority = age_priority
        self.slow_session_log_s = slow_session_log_s
        self._session_first_seen: dict[str, float] = {}
        # session id -> (calls, seconds inside model calls, prompt tokens,
        # completion tokens)
        self._session_activity: dict[str, tuple[int, float, int, int]] = {}
        self.tokenizer = tokenizer
        self.tokenizer_chat_template_kwargs = tokenizer_chat_template_kwargs
        self.exclude_tools_when_tool_choice_none = exclude_tools_when_tool_choice_none
        self.request_timeout_s = request_timeout_s
        self.max_model_len = max_model_len
        self.base_url: Optional[str] = None
        self.server: Any = None
        self.thread: Optional[threading.Thread] = None
        self._client_session: Any = None

    def start(self) -> str:
        """Start the wrapper in a background uvicorn thread."""
        import aiohttp
        import uvicorn
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.responses import JSONResponse, Response

        @asynccontextmanager
        async def lifespan(_: FastAPI):
            timeout = (
                aiohttp.ClientTimeout(total=self.request_timeout_s)
                if self.request_timeout_s is not None
                else aiohttp.ClientTimeout(total=None)
            )
            async with aiohttp.ClientSession(
                timeout=timeout,
                # Explicit: the default TCPConnector(limit=100) would cap the whole
                # rollout path at 100 concurrent requests to the frontend.
                connector=aiohttp.TCPConnector(limit=0),
            ) as session:
                self._client_session = session
                try:
                    yield
                finally:
                    self._client_session = None

        app = FastAPI(lifespan=lifespan)

        @app.get("/health")
        async def health() -> dict[str, str]:
            return {
                "status": "ok",
                "dynamo_frontend_base_url": self.dynamo_frontend_base_url,
            }

        @app.post("/v1/chat/completions")
        async def chat_completions(request: Request) -> Response:
            try:
                request_body = _loads(await request.body())
            except ValueError as e:
                raise HTTPException(status_code=400, detail="Invalid JSON body.") from e
            if not isinstance(request_body, dict):
                raise HTTPException(
                    status_code=400,
                    detail="Chat completion body must be a JSON object.",
                )

            try:
                prepared_body = await asyncio.to_thread(
                    prepare_dynamo_chat_completion_request,
                    request_body,
                    tokenizer=self.tokenizer,
                    tokenizer_chat_template_kwargs=self.tokenizer_chat_template_kwargs,
                    exclude_tools_when_tool_choice_none=self.exclude_tools_when_tool_choice_none,
                    max_model_len=self.max_model_len,
                )
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e

            session_id = pop_gym_conversation_id(prepared_body)
            if session_id and (self.age_priority or self.slow_session_log_s):
                first_seen = self._first_seen(session_id)
                if self.age_priority:
                    self._apply_age_priority(prepared_body, first_seen)
            call_start = time.time()
            status_code, response_body = await self._forward_chat_completion(
                prepared_body,
                authorization=request.headers.get("authorization"),
                session_id=session_id,
            )
            if session_id and self.slow_session_log_s:
                self._log_slow_session(session_id, call_start, response_body)
            if 200 <= status_code < 300:
                try:
                    _inject_gym_token_metadata(response_body)
                except ValueError as e:
                    return JSONResponse(
                        content={"error": {"message": str(e)}},
                        status_code=502,
                    )
            # Serialized here, once, as bytes: JSONResponse would run the stdlib
            # encoder over the whole response on this event loop.
            return Response(
                content=_dumps(response_body),
                status_code=status_code,
                media_type="application/json",
            )

        node_ip = _get_node_ip_local()
        free_port = _get_free_port_local(
            DEFAULT_DYNAMO_HTTP_PORT_RANGE_LOW,
            DEFAULT_DYNAMO_HTTP_PORT_RANGE_HIGH,
        )
        self.base_url = f"http://{node_ip}:{free_port}/v1"

        config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=free_port,
            timeout_keep_alive=120,
        )
        self.server = uvicorn.Server(config=config)
        self.thread = threading.Thread(
            target=self.server.run,
            name="dynamo-token-wrapper",
            daemon=True,
        )
        self.thread.start()
        return self.base_url

    def _first_seen(self, session_id: str) -> float:
        """Return the time of the first request of a session, noting it on first use."""
        first_seen = self._session_first_seen.get(session_id)
        if first_seen is None:
            first_seen = time.time()
            self._session_first_seen[session_id] = first_seen
            while len(self._session_first_seen) > _MAX_TRACKED_SESSIONS:
                # Insertion order: the oldest conversation goes first.
                del self._session_first_seen[next(iter(self._session_first_seen))]
        return first_seen

    def _apply_age_priority(
        self, request_body: dict[str, Any], first_seen: float
    ) -> None:
        """Set the agent-hint priority of a request from the age of its session."""
        nvext = request_body.setdefault("nvext", {})
        nvext.setdefault("agent_hints", {})["priority"] = age_priority(first_seen)

    def _log_slow_session(
        self, session_id: str, call_start: float, response_body: dict[str, Any]
    ) -> None:
        """Log a call of a session older than ``slow_session_log_s``.

        ``busy_s`` is the time the session spent inside model calls and ``idle_s`` the
        rest of its age (tools, Gym, queueing in the client), so the line shows whether
        a late conversation waits for the model or for something else.
        """
        now = time.time()
        usage = response_body.get("usage") or {}
        calls, busy_s, prompt_tokens, completion_tokens = self._session_activity.pop(
            session_id, (0, 0.0, 0, 0)
        )
        calls, busy_s = calls + 1, busy_s + (now - call_start)
        prompt_tokens += usage.get("prompt_tokens") or 0
        completion_tokens += usage.get("completion_tokens") or 0
        self._session_activity[session_id] = (
            calls,
            busy_s,
            prompt_tokens,
            completion_tokens,
        )
        while len(self._session_activity) > _MAX_TRACKED_SESSIONS:
            del self._session_activity[next(iter(self._session_activity))]
        age_s = now - self._session_first_seen[session_id]
        if age_s < self.slow_session_log_s:
            return
        print(
            f"[SESSION-SLOW] session={session_id[:12]} age_s={age_s:.0f} calls={calls} "
            f"busy_s={busy_s:.0f} idle_s={age_s - busy_s:.0f} "
            f"last_call_s={now - call_start:.1f} "
            f"prompt_tokens={usage.get('prompt_tokens')} "
            f"completion_tokens={usage.get('completion_tokens')} "
            f"total_prompt_tokens={prompt_tokens} "
            f"total_completion_tokens={completion_tokens}",
            flush=True,
        )

    async def _forward_chat_completion(
        self,
        request_body: dict[str, Any],
        *,
        authorization: Optional[str],
        session_id: Optional[str] = None,
    ) -> tuple[int, dict[str, Any]]:
        import aiohttp

        url = f"{self.dynamo_frontend_base_url.rstrip('/')}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if authorization:
            headers["Authorization"] = authorization
        if session_id:
            headers[DYNAMO_SESSION_ID_HEADER] = session_id

        session = self._client_session
        if session is None:
            return 503, {"error": {"message": "Dynamo token wrapper is not ready."}}
        try:
            async with session.post(
                url,
                data=_dumps(request_body),
                headers=headers,
            ) as response:
                raw_response = await response.read()
                if not raw_response:
                    return response.status, {}
                try:
                    response_body = _loads(raw_response)
                except ValueError:
                    response_body = {"raw": raw_response.decode(errors="replace")}
                if not isinstance(response_body, dict):
                    response_body = {"response": response_body}
                return response.status, response_body
        except asyncio.TimeoutError:
            return 504, {"error": {"message": f"Timed out forwarding to {url}."}}
        except aiohttp.ClientError as e:
            return 502, {
                "error": {
                    "message": f"Failed to forward request to {url}: {type(e).__name__}: {e}"
                }
            }

    def shutdown(self) -> None:
        """Stop the background uvicorn server."""
        if self.server is not None:
            self.server.should_exit = True
        if self.thread is not None:
            self.thread.join(timeout=10)
