import json
import shutil
from pathlib import Path
from zipfile import ZipFile, ZipInfo

import pytest

from cheatbench_inspect.openmath.build import RESOURCES, SITE_BUNDLE, write_archive

FIXTURES = Path(__file__).parent / "fixtures"
BLOG = "terrytao.wordpress.com"
API = "public-api.wordpress.com"
BACKGROUND = "/2026/07/21/a-digestion-of-the-jacobian-conjecture-counterexample/"
POSTS_API = f"/rest/v1.1/sites/{BLOG}/posts/"
CA_PATH = "/usr/local/share/ca-certificates/site-root.crt"
ROWS = ("goldbach_ce", "collatz_ce")
TITLES = ("Goldbach example", "Collatz example")
POSTS = ("/2026/08/03/example-a/", "/2026/07/29/example-b/")
ARCHIVES = ("/archive/example-a.zip", "/archive/example-b.zip")


@pytest.fixture
def task_dir(tmp_path: Path) -> Path:
    """A tiny synthetic capture with the original's layout and response types."""
    root = tmp_path / "data"
    root.mkdir()
    shutil.copyfile(FIXTURES / "rows.jsonl", root / "data.jsonl")
    site = tmp_path / "site"
    pages = site / "pages"
    pages.mkdir(parents=True)
    (pages / "background.html").write_text('<div class="post-content">Background.</div>')
    (pages / "missing.html").write_text("<title>Not Found</title>")
    (pages / "index__q_s_zzqqxyznoresults.html").write_text(
        '<div id="content"><h1>Not found</h1></div>'
    )
    manifest = [
        {
            "url": f"https://{BLOG}{path}",
            "path_component": path,
            "file": f"pages/{name}.html",
            "content_type": "text/html",
            "response_headers": {"content-type": "text/html"},
            "status": status,
        }
        for path, name, status in [(BACKGROUND, "background", 200), ("/missing", "missing", 404)]
    ]
    (site / "manifest.json").write_text(json.dumps(manifest))
    write_archive(root / SITE_BUNDLE, site, sorted(site.iterdir()))
    for folder in ("posts", "archives"):
        (root / f"files/{folder}").mkdir(parents=True)
    overrides: dict[str, dict] = {}
    entries = []
    for i, (row, title, post, material) in enumerate(
        zip(ROWS, TITLES, POSTS, ARCHIVES, strict=True), start=1
    ):
        html_file = f"files/posts/{row}.html"
        zip_file = f"files/archives/{row}.zip"
        (root / html_file).write_text(f'<h1>{title}</h1><a href="{material}">Materials</a>')
        with ZipFile(root / zip_file, "w") as archive:
            archive.writestr(ZipInfo("writeup.md"), f"Fixture content for {title}.")
        overrides[post] = {"status": 200, "content_type": "text/html", "body_file": html_file}
        overrides[post.rstrip("/")] = {
            "status": 301,
            "headers": {"location": f"https://{BLOG}{post}"},
        }
        overrides[material] = {
            "status": 200,
            "content_type": "application/zip",
            "body_file": zip_file,
        }
        date = post.strip("/")[:10].replace("/", "-")
        entries.append(
            {
                "task_id": row,
                "path": post,
                "file": html_file,
                "title_html": title,
                "title_text": title,
                "url": f"https://{BLOG}{post}",
                "post_id": str(18000 + i),
                "slug": post.strip("/").rsplit("/", 1)[1],
                "date_iso": f"{date}T12:00:00Z",
                "date_ymd": date,
                "date_str": date,
                "cats_html": "",
                "tags_html": "",
                "comments_label": "",
                "excerpt_html": title,
                "text": title,
                "shortlink": str(500 + i),
            }
        )
    (root / "overrides.json").write_text(json.dumps(overrides))
    (root / "overrides.index.json").write_text(json.dumps(entries))
    return root


@pytest.fixture
def mirror_code(tmp_path: Path) -> Path:
    directory = tmp_path / "server"
    directory.mkdir()
    for module in ("mirror", "inject"):
        shutil.copyfile(RESOURCES / f"{module}.txt", directory / f"{module}.py")
    return directory / "mirror.py"
