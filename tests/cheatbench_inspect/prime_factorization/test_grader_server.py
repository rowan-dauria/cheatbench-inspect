"""Exercise the grader's HTTP interface without Docker or a model."""

import json
from urllib.request import Request, urlopen

from inspect_ai.tool import ToolDef

from cheatbench_inspect.prime_factorization import verify_factors
from cheatbench_inspect.resources.prime_factorization.verifier import make_mcp_verifier
from cheatbench_inspect.resources.sandbox.grader_server import GraderServer


def test_http_client_discovers_and_calls_episode_verifier() -> None:
    schema = {
        "type": "object",
        "properties": {"p": {"type": "string"}, "q": {"type": "string"}},
        "required": ["p", "q"],
    }
    server = GraderServer(
        state={},
        call=make_mcp_verifier(15),
        server_name="verifier",
        tool_name="verify_factors",
        description=ToolDef(verify_factors(15)).description,
        schema=schema,
    )

    def rpc(request_id: int, method: str, params: dict) -> dict:
        request = Request(
            server.url,
            data=json.dumps(
                {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
            ).encode(),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        with urlopen(request, timeout=5) as response:
            reply = json.load(response)
        assert reply["id"] == request_id
        return reply["result"]

    server.start()
    try:
        assert rpc(1, "initialize", {})["serverInfo"]["name"] == "verifier"
        [tool] = rpc(2, "tools/list", {})["tools"]
        assert tool["name"] == "verify_factors"
        assert tool["inputSchema"] == schema
        for used, q in enumerate(["3", "5"], start=1):
            result = rpc(
                used + 2, "tools/call", {"name": tool["name"], "arguments": {"p": "3", "q": q}}
            )
            assert result["isError"] is False
            payload = json.loads(result["content"][0]["text"])
            assert payload["attempts_used"] == used
            assert payload["correct"] is (q == "5")
    finally:
        server.close()
