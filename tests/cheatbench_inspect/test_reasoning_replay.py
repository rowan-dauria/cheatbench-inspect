import asyncio
import base64
import html
import json
import re
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from inspect_ai import Task, eval
from inspect_ai.agent import sandbox_agent_bridge
from inspect_ai.agent._bridge.responses_impl import (
    messages_from_responses_input,
    responses_output_items_from_assistant_message,
)
from inspect_ai.dataset import Sample
from inspect_ai.model import (
    ChatCompletionChoice,
    ChatMessageAssistant,
    ChatMessageUser,
    ContentReasoning,
    ContentText,
    GenerateConfig,
    ModelAPI,
    ModelOutput,
    ModelUsage,
    modelapi,
)
from inspect_ai.model._openai_responses import openai_responses_inputs
from inspect_ai.model._openrouter_reasoning import openrouter_reasoning_details_to_reasoning
from inspect_ai.model._providers.openrouter import OpenRouterAPI
from inspect_ai.solver import solver
from inspect_ai.tool import ToolCall

from cheatbench_inspect.reasoning_replay import (
    RESPONSES_REASONING_MARKER,
    anthropic_reasoning_filter,
    pack_reasoning,
    responses_reasoning_filter,
    unpack_reasoning,
)
from cheatbench_inspect.resources.codex import proxy as codex_proxy

DETAILS = [
    [{"type": "reasoning.text", "text": "plain thought", "format": "unknown"}],
    [
        {"type": "reasoning.encrypted", "data": "cipher"},
        {"type": "reasoning.summary", "summary": "readable"},
    ],
]
FORBIDDEN_SANDBOX_TERMS = ("cheatbench", "inspect", "eval")
records = []
original = [ContentReasoning(reasoning="placeholder")]


@modelapi(name="cheatbench_recording")
class Recorder(ModelAPI):
    async def generate(self, input, tools, tool_choice, config):
        records.append(input)
        if any(m.role == "tool" for m in input):
            message = ChatMessageAssistant(content="done")
            stop = "stop"
        else:
            message = ChatMessageAssistant(
                content=[original[0], ContentText(text="hello")],
                tool_calls=[ToolCall(id="call_1", function="test", arguments={"x": "y"})],
            )
            stop = "tool_calls"
        return ModelOutput(
            model="cheatbench_recording/test",
            choices=[ChatCompletionChoice(message=message, stop_reason=stop)],
            usage=ModelUsage(input_tokens=10, output_tokens=5, total_tokens=15),
        )


@pytest.mark.parametrize("details", DETAILS)
def test_round_trip(details):
    content = openrouter_reasoning_details_to_reasoning(details)
    assert unpack_reasoning(pack_reasoning(content)) == content


def test_foreign_envelopes():
    legacy = base64.b64encode(
        json.dumps(
            {
                "type": "reasoning.text",
                "text": "s",
                "cheatbench_inspect_content_reasoning": {"reasoning": "x"},
            }
        ).encode("utf-8")
    ).decode("ascii")
    for value in [
        "plain",
        "openrouter.reasoning:bad!",
        "openrouter.reasoning:e30=",
        f"openrouter.reasoning:{legacy}",
    ]:
        content = ContentReasoning(reasoning=value)
        assert unpack_reasoning(content) == content


@pytest.mark.parametrize("details", DETAILS)
async def test_filters_and_proxy_send_no_identifying_strings(details):
    """Decode everything sent to sandboxed CLIs and assert no identifying terms appear."""
    # 1. Anthropic filter envelope (base64 signature sent to Claude Code).
    reasoning = openrouter_reasoning_details_to_reasoning(details)
    packed_anthropic = pack_reasoning(reasoning)
    raw_sig = packed_anthropic.reasoning
    assert raw_sig.startswith("openrouter.reasoning:")
    decoded_anthropic = base64.b64decode(raw_sig.removeprefix("openrouter.reasoning:")).decode(
        "utf-8"
    )
    for blob in (raw_sig, decoded_anthropic):
        lower = blob.lower()
        for term in FORBIDDEN_SANDBOX_TERMS:
            assert term not in lower, f"found {term!r} in Anthropic envelope: {blob!r}"

    # 2. Responses filter <think internal=...> attribute sent to the bridge / Codex proxy.
    turn_output = ModelOutput.from_message(
        ChatMessageAssistant(
            content=[
                ContentReasoning(
                    reasoning="gAAAA_ENCRYPTED",
                    summary="Summary text",
                    signature="rs_anon_1",
                    redacted=True,
                ),
                ContentText(
                    text="Reply",
                    internal={"message_id": "msg_anon_1", "message_phase": "final_answer"},
                ),
            ],
            model="openai/gpt-5.6-luna",
        ),
        stop_reason="stop",
    )

    class _StubModel:
        async def generate(self, input, tools=None, tool_choice=None, config=None):
            return turn_output

    filt = responses_reasoning_filter()
    filtered_out = await filt(
        _StubModel(),  # type: ignore[arg-type]
        [ChatMessageUser(content="hi")],
        [],
        None,
        GenerateConfig(),
    )
    assert isinstance(filtered_out, ModelOutput)
    wire_items = [
        item.model_dump(mode="json", exclude_none=True)
        for item in responses_output_items_from_assistant_message(filtered_out.message)
    ]
    part_texts = [
        part["text"]
        for item in wire_items
        for part in item.get("content", [])
        if isinstance(part, dict) and isinstance(part.get("text"), str)
    ]
    internal_attrs = [
        match for text in part_texts for match in re.findall(r'internal="([^"]*)"', text)
    ]
    assert internal_attrs, "expected <think internal=...> attribute in bridge output"
    for encoded_attr in internal_attrs:
        decoded_attr = base64.b64decode(html.unescape(encoded_attr)).decode("utf-8")
        assert RESPONSES_REASONING_MARKER in json.loads(decoded_attr)
        for blob in (encoded_attr, decoded_attr):
            lower = blob.lower()
            for term in FORBIDDEN_SANDBOX_TERMS:
                assert term not in lower, f"found {term!r} in Responses <think internal>: {blob!r}"

    # 3. Also check rewritten output sent to Codex and proxy.py source itself.
    rewritten = codex_proxy.rewrite_response_payload({"output": wire_items})
    rewritten_json = json.dumps(rewritten).lower()
    proxy_source = await asyncio.to_thread(Path(codex_proxy.__file__).read_text, encoding="utf-8")
    proxy_source_lower = proxy_source.lower()
    for term in FORBIDDEN_SANDBOX_TERMS:
        assert term not in rewritten_json
        assert term not in proxy_source_lower


async def test_responses_filter_roundtrip_all_cases():
    """Verify encrypted-only, summary+encrypted, missing id, edge tags, and tool calls."""
    r1 = ContentReasoning(
        reasoning="gAAAA_ENC_1",
        summary=None,
        signature="rs_1",
        redacted=True,
    )
    r2 = ContentReasoning(
        reasoning="gAAAA_ENC_2",
        summary="Edge </summary> and <think>text</think> inside summary.",
        signature="rs_2",
        redacted=True,
    )
    prose = ContentText(
        text="Model prose quoting <think>literal</think> safely.",
        internal={"message_id": "msg_commentary_1", "message_phase": "commentary"},
    )
    tc_custom = ToolCall(
        id="call_c1", function="exec", arguments={"input": "echo hi"}, type="custom"
    )
    tc_func = ToolCall(id="call_f1", function="wait", arguments={"cell_id": "abc"}, type="function")
    turn1_output = ModelOutput.from_message(
        ChatMessageAssistant(
            content=[r1, r2, prose],
            tool_calls=[tc_custom, tc_func],
            model="openai/gpt-5.6-luna",
        ),
        stop_reason="tool_calls",
    )

    class _StubModel:
        def __init__(self) -> None:
            self.seen_inputs: list[list[Any]] = []

        async def generate(self, input, tools=None, tool_choice=None, config=None):
            self.seen_inputs.append(input)
            return turn1_output

    stub = _StubModel()
    filt = responses_reasoning_filter()

    out1 = await filt(
        stub,  # type: ignore[arg-type]
        [ChatMessageUser(content="go")],
        [],
        None,
        GenerateConfig(),
    )
    assert isinstance(out1, ModelOutput)

    raw_items = [
        item.model_dump(mode="json", exclude_none=True)
        for item in responses_output_items_from_assistant_message(
            out1.message,
            tool_namespaces={"exec": "functions", "wait": "functions"},
        )
    ]
    assert len(raw_items) == 5

    rewritten = codex_proxy.rewrite_response_payload(
        {"id": "resp_1", "object": "response", "status": "completed", "output": raw_items}
    )
    out_items = rewritten["output"]
    assert out_items[0]["type"] == "reasoning"
    assert out_items[0]["id"] == "rs_1"
    assert out_items[0]["encrypted_content"] == "gAAAA_ENC_1"
    assert out_items[0]["summary"] == []

    assert out_items[1]["type"] == "reasoning"
    assert out_items[1]["id"] == "rs_2"
    assert out_items[1]["encrypted_content"] == "gAAAA_ENC_2"
    assert out_items[1]["summary"] == [
        {
            "type": "summary_text",
            "text": "Edge </summary> and <think>text</think> inside summary.",
        }
    ]

    # Assistant prose has <content-internal> stripped and native id/phase set.
    assert out_items[2]["type"] == "message"
    assert out_items[2]["id"] == "msg_commentary_1"
    assert out_items[2]["phase"] == "commentary"
    assert (
        out_items[2]["content"][0]["text"] == "Model prose quoting <think>literal</think> safely."
    )

    # Tool calls intact with namespaces preserved.
    assert out_items[3]["type"] == "custom_tool_call"
    assert out_items[3]["name"] == "exec"
    assert out_items[3]["namespace"] == "functions"
    assert out_items[4]["type"] == "function_call"
    assert out_items[4]["name"] == "wait"
    assert out_items[4]["namespace"] == "functions"

    # Simulate Turn 2 where Codex omits `id` on reasoning items under store=False,
    # and echoes native `id` and `phase` on the assistant message.
    turn2_codex_req = {
        "model": "openai/gpt-5.6-luna",
        "store": False,
        "input": [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "go"}]},
            {
                "type": "reasoning",
                "summary": [],
                "content": None,
                "encrypted_content": "gAAAA_ENC_1",
            },
            {
                "type": "reasoning",
                "summary": [
                    {
                        "type": "summary_text",
                        "text": "Edge </summary> and <think>text</think> inside summary.",
                    }
                ],
                "content": None,
                "encrypted_content": "gAAAA_ENC_2",
            },
            out_items[2],
            out_items[3],
            out_items[4],
            {"type": "custom_tool_call_output", "call_id": "call_c1", "output": "hi\n"},
            {"type": "function_call_output", "call_id": "call_f1", "output": "done"},
        ],
    }

    bridge_req = codex_proxy.rewrite_request_payload(turn2_codex_req)
    turn2_messages = messages_from_responses_input(bridge_req["input"], [], "openai/gpt-5.6-luna")
    await filt(stub, turn2_messages, [], None, GenerateConfig())  # type: ignore[arg-type]
    restored_turn2_messages = stub.seen_inputs[-1]

    wire_items = [
        cast(dict[str, Any], it) for it in await openai_responses_inputs(restored_turn2_messages)
    ]
    wire_reasoning = [
        it for it in wire_items if isinstance(it, dict) and it.get("type") == "reasoning"
    ]
    assert wire_reasoning[0]["id"] == "rs_1"
    assert wire_reasoning[0]["encrypted_content"] == "gAAAA_ENC_1"
    assert wire_reasoning[0]["summary"] == []
    assert wire_reasoning[0]["content"] == []

    assert wire_reasoning[1]["id"] == "rs_2"
    assert wire_reasoning[1]["encrypted_content"] == "gAAAA_ENC_2"
    assert wire_reasoning[1]["summary"] == [
        {
            "type": "summary_text",
            "text": "Edge </summary> and <think>text</think> inside summary.",
        }
    ]
    assert wire_reasoning[1]["content"] == []

    wire_asst_msgs = [
        it
        for it in wire_items
        if isinstance(it, dict) and it.get("type") == "message" and it.get("role") == "assistant"
    ]
    assert len(wire_asst_msgs) == 1
    assert wire_asst_msgs[0]["id"] == "msg_commentary_1"
    assert wire_asst_msgs[0]["phase"] == "commentary"

    # Without the filter, Inspect drops summary when redacted=True and cannot restore omitted id.
    baseline_wire = [
        cast(dict[str, Any], it) for it in await openai_responses_inputs(turn2_messages)
    ]
    baseline_reasoning = [
        it for it in baseline_wire if isinstance(it, dict) and it.get("type") == "reasoning"
    ]
    assert baseline_reasoning[0].get("id") is None
    assert baseline_reasoning[1].get("id") is None
    assert baseline_reasoning[1]["summary"] == []


async def test_responses_filter_instances_have_isolated_caches():
    """Each responses_reasoning_filter() call returns a fresh per-bridge closure."""
    turn_output = ModelOutput.from_message(
        ChatMessageAssistant(
            content=[
                ContentReasoning(
                    reasoning="gAAAA_SHARED_BLOB",
                    summary="Secret from bridge 1",
                    signature="rs_bridge_1",
                    redacted=True,
                )
            ]
        ),
        stop_reason="stop",
    )

    class _StubModel:
        def __init__(self) -> None:
            self.seen_inputs: list[list[Any]] = []

        async def generate(self, input, tools=None, tool_choice=None, config=None):
            self.seen_inputs.append(input)
            return turn_output

    stub1, stub2 = _StubModel(), _StubModel()
    filt1 = responses_reasoning_filter()
    filt2 = responses_reasoning_filter()

    await filt1(stub1, [ChatMessageUser(content="t1")], [], None, GenerateConfig())  # type: ignore[arg-type]

    # Bridge 2 sees the same encrypted blob without signature or summary: must not see bridge 1's cache.
    inbound = [
        ChatMessageAssistant(
            content=[ContentReasoning(reasoning="gAAAA_SHARED_BLOB", redacted=True)]
        )
    ]
    await filt2(stub2, inbound, [], None, GenerateConfig())  # type: ignore[arg-type]
    replayed_2 = stub2.seen_inputs[-1][0].content[0]
    assert replayed_2.signature is None
    assert replayed_2.summary is None

    # Bridge 1 restores its own cached signature and summary.
    await filt1(stub1, inbound, [], None, GenerateConfig())  # type: ignore[arg-type]
    replayed_1 = stub1.seen_inputs[-1][0].content[0]
    assert replayed_1.signature == "rs_bridge_1"
    assert replayed_1.summary == "Secret from bridge 1"


async def test_responses_filter_unpacks_untranslated_marker_and_plaintext():
    """Verify fallback when a packed item returns without proxy translation or is plaintext."""
    packed = ContentReasoning(
        reasoning="gAAAA_FALLBACK",
        summary="Fallback summary",
        signature="rs_fb",
        redacted=True,
        internal={
            RESPONSES_REASONING_MARKER: {
                "id": "rs_fb",
                "summary": "Fallback summary",
                "encrypted_content": "gAAAA_FALLBACK",
                "content": None,
            }
        },
    )
    plaintext = ContentReasoning(
        reasoning="Plain thought",
        signature="rs_plain",
        redacted=False,
        internal={
            RESPONSES_REASONING_MARKER: {
                "id": "rs_plain",
                "summary": None,
                "encrypted_content": None,
                "content": "Plain thought",
            },
            "keep": 1,
        },
    )

    class _StubModel:
        def __init__(self) -> None:
            self.seen_inputs: list[list[Any]] = []

        async def generate(self, input, tools=None, tool_choice=None, config=None):
            self.seen_inputs.append(input)
            return ModelOutput.from_content("stub/model", "ok")

    stub = _StubModel()
    filt = responses_reasoning_filter()
    await filt(
        stub,  # type: ignore[arg-type]
        [ChatMessageAssistant(content=[packed, plaintext])],
        [],
        None,
        GenerateConfig(),
    )
    out_fb, out_plain = stub.seen_inputs[-1][0].content
    assert out_fb.signature == "rs_fb"
    assert out_fb.summary == "Fallback summary"
    assert out_fb.redacted is False
    assert out_fb.internal == {"reasoning_encrypted_content": "gAAAA_FALLBACK"}

    assert out_plain.reasoning == "Plain thought"
    assert out_plain.signature == "rs_plain"
    assert out_plain.internal == {"keep": 1}


@pytest.mark.slow
@pytest.mark.parametrize("details", DETAILS)
@pytest.mark.parametrize("stream", [False, True])
def test_bridge(details, stream, tmp_path):
    original[0] = openrouter_reasoning_details_to_reasoning(details)
    results = []

    @solver
    def run(filtered):
        async def solve(state, generate):
            async with (
                sandbox_agent_bridge(
                    filter=anthropic_reasoning_filter if filtered else None
                ) as bridge,
                httpx.AsyncClient(timeout=30) as client,
            ):
                payload = {
                    "model": "cheatbench_recording/test",
                    "max_tokens": 100,
                    "stream": stream,
                    "messages": [{"role": "user", "content": "hi"}],
                    "tools": [
                        {
                            "name": "test",
                            "description": "test",
                            "input_schema": {
                                "type": "object",
                                "properties": {"x": {"type": "string"}},
                            },
                        }
                    ],
                }

                async def post():
                    for attempt in range(100):
                        try:
                            response = await client.post(
                                f"http://127.0.0.1:{bridge.port}/v1/messages",
                                headers={
                                    "x-api-key": "test",
                                    "anthropic-version": "2023-06-01",
                                },
                                json=payload,
                            )
                            response.raise_for_status()
                            return response
                        except httpx.ConnectError:
                            if attempt == 99:
                                raise
                            await asyncio.sleep(0.1)
                    raise AssertionError("unreachable")

                response = await post()
                if stream:
                    blocks = []
                    metadata = {}
                    for line in response.text.splitlines():
                        if not line.startswith("data: "):
                            continue
                        event = json.loads(line[6:])
                        if event["type"] == "message_start":
                            metadata["usage"] = event["message"]["usage"]
                        if event["type"] == "message_delta":
                            metadata["stop_reason"] = event["delta"]["stop_reason"]
                            metadata["usage"].update(event["usage"])
                        if event["type"] == "content_block_start":
                            blocks.append(event["content_block"])
                        if event["type"] == "content_block_delta":
                            block = blocks[event["index"]]
                            delta = event["delta"]
                            if delta["type"] == "text_delta":
                                block["text"] += delta["text"]
                            elif delta["type"] == "thinking_delta":
                                block["thinking"] += delta["thinking"]
                            elif delta["type"] == "signature_delta":
                                block["signature"] = block.get("signature", "") + delta["signature"]
                            elif delta["type"] == "input_json_delta":
                                block["_partial"] = (
                                    block.get("_partial", "") + delta["partial_json"]
                                )
                    for block in blocks:
                        if "_partial" in block:
                            block["input"] = json.loads(block.pop("_partial"))
                else:
                    body = response.json()
                    blocks = body["content"]
                    metadata = {k: body[k] for k in ("usage", "stop_reason")}
                payload["messages"] += [
                    {"role": "assistant", "content": blocks},
                    {
                        "role": "user",
                        "content": [
                            {"type": "tool_result", "tool_use_id": "call_1", "content": "ok"}
                        ],
                    },
                ]
                await post()
                if filtered:
                    received = [
                        c
                        for m in records[-1]
                        if m.role == "assistant" and isinstance(m.content, list)
                        for c in m.content
                        if isinstance(c, ContentReasoning)
                    ]
                    assert received == [original[0]]
                    upstream = await OpenRouterAPI(
                        "example/model", api_key="offline"
                    ).messages_to_openai(records[-1])
                    assert dict(upstream[1]).get("reasoning_details") == details
                    assert any(b["type"] == "thinking" for b in blocks)
                    assert not any(
                        b.get("type") == "text" and "thought" in b.get("text", "") for b in blocks
                    )
                results.append((blocks, metadata))
            return state

        return solve

    for filtered in (False, True):
        records.clear()
        logs = eval(
            Task(dataset=[Sample(input="hi")], solver=run(filtered), sandbox="local"),
            model="cheatbench_recording/test",
            display="none",
            log_dir=str(tmp_path),
        )
        assert logs[0].status == "success", logs[0].error
    baseline, filtered_result = results
    assert baseline[1] == filtered_result[1]
    assert [b for b in baseline[0] if b["type"] == "tool_use"] == [
        b for b in filtered_result[0] if b["type"] == "tool_use"
    ]
    assert [b["text"] for b in filtered_result[0] if b["type"] == "text"] == ["hello"]
