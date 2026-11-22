import asyncio
import json

import httpx
import pytest
from inspect_ai import Task, eval
from inspect_ai.agent import sandbox_agent_bridge
from inspect_ai.dataset import Sample
from inspect_ai.model import (
    ChatCompletionChoice,
    ChatMessageAssistant,
    ContentReasoning,
    ContentText,
    ModelAPI,
    ModelOutput,
    ModelUsage,
    modelapi,
)
from inspect_ai.model._openrouter_reasoning import openrouter_reasoning_details_to_reasoning
from inspect_ai.model._providers.openrouter import OpenRouterAPI
from inspect_ai.solver import solver
from inspect_ai.tool import ToolCall

from cheatbench_inspect.reasoning_replay import (
    anthropic_reasoning_filter,
    pack_reasoning,
    unpack_reasoning,
)

DETAILS = [
    [{"type": "reasoning.text", "text": "plain thought", "format": "unknown"}],
    [
        {"type": "reasoning.encrypted", "data": "cipher"},
        {"type": "reasoning.summary", "summary": "readable"},
    ],
]
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
    for value in ["plain", "openrouter.reasoning:bad!", "openrouter.reasoning:e30="]:
        content = ContentReasoning(reasoning=value)
        assert unpack_reasoning(content) == content


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
