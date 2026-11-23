"""Hand the agent's staged files to its user without touching system directories."""

from cheatbench_inspect.harnesses.container import BOOTSTRAP, agent_owned_roots


def test_staged_homes_and_workspace_go_to_the_agent() -> None:
    staged = [
        "/workspace/instance.txt",
        "/home/runner/work/factoring-evals/factoring-evals/README.md",
        "/home/runner/work/_temp/generation/gen-20260824-060d.log",
        "/etc/motd",
    ]

    assert agent_owned_roots(staged) == ["/etc/motd", "/home/runner", "/workspace"]


def test_launcher_names_nothing_from_the_eval() -> None:
    for word in ("verif", "grader", "factor", "terminus", "harbor", "cheat"):
        assert word not in BOOTSTRAP.lower()
