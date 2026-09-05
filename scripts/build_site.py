"""Package the static page with content-versioned asset URLs for GitHub Pages."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def build_site(source: Path, destination: Path) -> None:
    """Keep cached HTML usable while giving changed assets fresh cache keys."""
    html = (source / "index.html").read_text(encoding="utf-8")
    for name in ("styles.css", "favicon.svg"):
        version = hashlib.sha256((source / name).read_bytes()).hexdigest()[:16]
        reference = f'href="./{name}"'
        if html.count(reference) != 1:
            raise ValueError(f"Expected exactly one reference to {name}")
        html = html.replace(reference, f'href="./{name}?v={version}"')

    shutil.copytree(source, destination, dirs_exist_ok=True)
    (destination / "index.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    output = _ROOT / "dist" / "site"
    build_site(_ROOT / "site", output)
    print(f"Built cache-versioned site in {output.relative_to(_ROOT)}")
