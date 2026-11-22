"""Claude Agent SDK adapter for the prime factorization verifier."""

import json
import re

from claude_agent_sdk import create_sdk_mcp_server, tool  # type: ignore[import-not-found]  # inspect-evals-lint: ignore[external_dependencies] -- Installed in the Claude sandbox image.

from .verifier import make_state_verifier

DESCRIPTION = (
    "Check one proposed factor pair against the number. Attempts allowed: 50. "
    "Provide p and q as decimal strings. Returns only boolean validity and "
    "correctness plus attempt counts; it never reveals the product, a quotient, "
    "a remainder, or either factor. Invalid candidates do not consume checks. A "
    "correct result accepts your answer; you can then finish."
)
INPUT_SCHEMA = {
    "type": "object",
    "properties": {"p": {"type": "string"}, "q": {"type": "string"}},
    "required": ["p", "q"],
}


def make_grader(row: dict, staged: dict, workdir: str):
    """Build the original SDK server configuration and live episode state."""
    number = str(row["number"]).strip()
    if re.fullmatch(r"[0-9]+", number) is None:
        raise ValueError("the instance number must be a positive integer above 3")
    state, verifier = make_state_verifier(int(number), row["id"])

    @tool("verify_factors", DESCRIPTION, INPUT_SCHEMA)
    async def verify_factors(args):
        result = verifier(args.get("p"), args.get("q"))
        return {
            "content": [{"type": "text", "text": json.dumps(result)}],
            "is_error": not result["valid"],
        }

    server = create_sdk_mcp_server(name="verifier", version="2.0.0", tools=[verify_factors])
    return {"verifier": server}, ["mcp__verifier__verify_factors"], state
