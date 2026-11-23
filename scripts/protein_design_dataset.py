"""Build the pinned CheatBench Protein Design data and optionally publish it.

uv run python scripts/protein_design_dataset.py OUT_DIR
uv run python scripts/protein_design_dataset.py OUT_DIR --upload-existing REPO_ID
"""

import argparse
import hashlib
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

from geoguessr_dataset import UPSTREAM_COMMIT, fetch_verified, pack

TASK = "tasks/protein_design"
ARCHIVE_NAME = "protein_design.tar.gz"
FASTA_URL = "https://www.rcsb.org/fasta/entry/{pdb}"
UPSTREAM_FILES = {
    f"{TASK}/build.py": "64a0fd31e1acf576c60567f06359b3e415360c972682d7d71d4c98ece0c8b3eb",
    f"{TASK}/targets.json": "b0f83c9b26c8f6c26d7a401410c9501119c171c14d4bd750797d52a3a504a5db",
    "LICENSE": "05424bacbf4ec79bf8c2c8a0bf0eaf0f7b1b74379df933cb45b20b4b31248a22",
}
# Pin the complete FASTA records, including the selected headers.
FASTA_SHA256 = {
    "1AG6": "40835980b59592f83075438ba9fd7cf1b47cf4503f31fe107425299e32b8ae98",
    "1AMY": "6ce6584dca1a2ff02e3f7c3647d710ee78d333cc5726142894bda7392be6863b",
    "1APX": "b10753bcafeb844bc0de9e5d101f59795a9e4d9429676dd94ed5ccf92c3ad508",
    "1CHW": "6fe7b168e9dbea00dc596d74657563050460ae2c83642f323a31dadf1a565660",
    "1EKJ": "7c72ec03b8207aa42098d2ca68ecde0a0d8d75604087bdbffee00cf520d68de9",
    "1EYP": "81d4eadefd23b615d1ccebb22291f5ecf7715051279d11ae6944b5c0fda166c0",
    "1FND": "183699c662c74c1dd03e34267c6045ee8893b8a90043972c9ceca9243b3bec04",
    "1ITZ": "301c22004abce51d85c98ee5d01c3f17a553bf60c3e8386da4c7e1cf0ea05c37",
    "1SM4": "7657edb3bde672d10b2bb5cb8a70f5474bdd03c981edbcbbaa69ddda340ba3aa",
    "1SPI": "fbe031a11d96e09c7f0e57afd4483244f71739c8181b4aa5c7da5800b29c098f",
    "1Z90": "b21f394d02223338bc03069ae9d67bb6754598be94d8736c3c95342f23f7a446",
    "2XFR": "7b02e23c0e6c18565b4517705c8d63b9d610dfb4bb6a82c4f95c50b6acbb56d5",
    "3NWJ": "2aa4d06f19753ce308b40e04600bf863058adff5edaf56d54858fb388e5e783f",
    "4AIO": "964005b2e05bb483ac071b7d730ff730fa640a9de2da655999e77d857b5441d1",
    "4OHQ": "93484042051156c4ea2142098956cef95c9b73eee2b6828fce92c46a5a11afc5",
    "5NUF": "c44d6be72de9d8be8a88bc807d9b4c07010ecb00ba14b93cf674bc3a4ffc4f5f",
    "6KEX": "af46e268c28eb68e56e5779cb0bfdf7ffc58c30c67717ad60ce951860d6b7846",
    "7Q55": "3ac41adaca865341ede61079a0014d0fe4699eff247684a996d4d1833f3dffdb",
    "8RUC": "857a453246f8863573dbc9edc669facebc65168ef232486550d6eb66c051ee2c",
    "9TXO": "b3f6fe56201404296495bff2c72f4f6543a4e422e9d7d12c86b5a564859a3901",
}

# The original builder only needs AGENT_HOME from the sandbox runtime. All its
# writes are relative to the scratch copy of the task.
BUILD = """
import runpy
import sys
import types

sys.modules["core"] = types.ModuleType("core")
sandbox = types.ModuleType("core.sandbox")
sandbox.AGENT_HOME = "/home/user"
sys.modules["core.sandbox"] = sandbox
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
"""

DATASET_CARD = f"""---
license: other
pretty_name: CheatBench Protein Design (generated)
---

# CheatBench Protein Design

Twenty plant target rows and their staged files, generated with the unmodified
[CheatBench builder](https://github.com/centerforaisafety/cheatbench/tree/{UPSTREAM_COMMIT}/{TASK})
at its default seed. The benchmark materials and synthetic answer sequences are
released under the MIT license, included in the archive. The checker performs a
synthetic membership test for a behavioral evaluation.

The selected FASTA records come from the RCSB Protein Data Bank and are covered by
the [CC0 1.0 PDB data usage policy](https://www.rcsb.org/pages/usage-policy).
See the `targets.json` in the archive for the PDB entry and entity of each target.
"""


def checked_bytes(name: str, data: bytes, expected: str) -> bytes:
    digest = hashlib.sha256(data).hexdigest()
    if digest != expected:
        raise ValueError(f"{name} has sha256 {digest}, expected {expected}")
    return data


def canonical_fasta(data: bytes) -> bytes:
    # RCSB can return the same entities in a different order. The upstream
    # builder selects by entity identifier and preserves the selected header.
    records = data.strip().removeprefix(b">").split(b"\n>")
    return b">" + b"\n>".join(sorted(records)) + b"\n"


def build(out_dir: Path, upstream: Path | None, fasta_dir: Path | None) -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        task = root / TASK
        cache = task / ".cache"
        cache.mkdir(parents=True)
        for name, digest in UPSTREAM_FILES.items():
            data = (
                fetch_verified(name, digest)
                if upstream is None
                else checked_bytes(name, (upstream / name).read_bytes(), digest)
            )
            (root / name).write_bytes(data)
        for pdb, digest in FASTA_SHA256.items():
            filename = f"{pdb}.fasta"
            if fasta_dir is None:
                with urllib.request.urlopen(FASTA_URL.format(pdb=pdb), timeout=120) as response:
                    data = response.read()
            else:
                data = (fasta_dir / filename).read_bytes()
            (cache / filename).write_bytes(checked_bytes(filename, canonical_fasta(data), digest))
        subprocess.run(
            [sys.executable, "-I", "-B", "-c", BUILD, str(task / "build.py")],
            check=True,
            cwd=root,
        )
        shutil.copyfile(root / "LICENSE", task / "LICENSE")
        archive = out_dir / ARCHIVE_NAME
        digest = pack(task, archive, root_files=("data.jsonl", "targets.json", "LICENSE"))
        print(f"{archive}: sha256 {digest}")


def upload(repo_id: str, archive: Path) -> None:
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=False, exist_ok=True)
    commit = api.create_commit(
        repo_id,
        repo_type="dataset",
        operations=[
            CommitOperationAdd(ARCHIVE_NAME, str(archive)),
            CommitOperationAdd("README.md", DATASET_CARD.encode()),
        ],
        commit_message=f"Protein Design data from CheatBench {UPSTREAM_COMMIT[:7]}",
    )
    print(
        f"{repo_id} revision {commit.oid}, sha256 {hashlib.sha256(archive.read_bytes()).hexdigest()}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--upstream", type=Path, help="Read pinned sources from a local checkout.")
    parser.add_argument("--fasta-dir", type=Path, help="Reuse already downloaded pinned FASTAs.")
    parser.add_argument("--upload-existing", metavar="REPO_ID")
    args = parser.parse_args()
    if args.upload_existing:
        upload(args.upload_existing, args.out_dir / ARCHIVE_NAME)
    else:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        build(args.out_dir, args.upstream, args.fasta_dir)


if __name__ == "__main__":
    main()
